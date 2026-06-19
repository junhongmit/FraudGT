import logging
import os, time
import os.path as osp
from functools import partial
from typing import Union

import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from torch_geometric.data import (HeteroData, InMemoryDataset)
from ogb.lsc import MAG240MDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (DBLP, IMDB, OGB_MAG, Planetoid, MovieLens)
from fraudGT.datasets.aml_dataset import AMLDataset
from fraudGT.datasets.eth_dataset import ETHDataset
from fraudGT.datasets.fraud_review_dataset import FraudReviewDataset
from fraudGT.datasets.temporal_dataset import TemporalDataset
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from fraudGT.graphgym.register import register_loader
from fraudGT.transform.posenc_stats import compute_posenc_stats
from fraudGT.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size)


from torch_geometric.utils import (index_to_mask, to_undirected)
from fraudGT.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from fraudGT.loader.encoding_generator import (preprocess_Node2Vec, check_Node2Vec, load_Node2Vec,
                                                 preprocess_Metapath, check_Metapath, load_Metapath,\
                                                 preprocess_KGE, check_KGE, load_KGE)

# For large-scale heterophilic dataset download
dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}

def get_sparse_tensor(edge_index, num_nodes=None, num_src_nodes=None, num_dst_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t


    if (num_nodes is None) and (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes = int(edge_index.max()) + 1
        num_dst_nodes = num_src_nodes
    elif (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes, num_dst_nodes = num_nodes, num_nodes


    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_src_nodes, num_dst_nodes)).t()


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    # logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'num_nodes_dict'):
        total_num_nodes = sum(dataset.data.num_nodes_dict.values())
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )

@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'DBLP':
            dataset = preformat_DBLP(dataset_dir)

        elif pyg_dataset_id == 'IMDB':
            dataset = preformat_IMDB(dataset_dir)
        
        elif pyg_dataset_id == 'Planetoid':
            dataset = preformat_Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'MovieLens':
            dataset = preformat_MovieLens(dataset_dir)
        
        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbn') or name in ['MAG240M', 'mag-year']:
            dataset = preformat_OGB_Node(dataset_dir, name.replace('_', '-'))

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")

    elif format == 'AML':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_AML(dataset_dir, name)

    elif format == 'ETH':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_ETH(dataset_dir)

    elif format == 'FraudReview':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = FraudReviewDataset(root=dataset_dir, name=name)

    else:
        raise ValueError(f"Unknown data format: {format}")

    # pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute structural encodings
    if cfg.posenc_Hetero_Node2Vec.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Node2Vec(pe_dir):
            preprocess_Node2Vec(pe_dir, dataset)
        
        model = load_Node2Vec(pe_dir)
        if len(dataset) == 1:
            homo_data = dataset[0].to_homogeneous()
            for idx, node_type in enumerate(dataset.data.node_types):
                mask = homo_data.node_type == idx
                dataset.data[node_type]['pestat_Hetero_Node2Vec'] = model[mask]
        elif isinstance(dataset, TemporalDataset):
            for split_idx, split in enumerate(['train', 'val', 'test']):
                homo_data = dataset[split].to_homogeneous()
                for idx, node_type in enumerate(dataset[split].node_types):
                    mask = homo_data.node_type == idx
                    dataset[split][node_type]['pestat_Hetero_Node2Vec'] = model[split_idx][mask]
        else:
            data_list = []
            for idx in range(len(dataset)):
                data = dataset.get(idx)
                data.pestat_Hetero_Node2Vec = model[idx]
                data_list.append(data)
                # if idx >= 10:
                #     break
            data_list = list(filter(None, data_list))

            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

    if cfg.posenc_Hetero_Metapath.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Metapath(pe_dir):
            preprocess_Metapath(pe_dir, dataset)
        
        model = load_Metapath(pe_dir)
        emb = model['model'].weight.data.detach().cpu()
        if hasattr(dataset, 'dynamicTemporal'):
            data_list = []
            for split in range(3):
                data = dataset[split]
                for node_type in dataset.data.node_types:
                    data[node_type]['pestat_Hetero_Metapath'] = \
                        emb[model['start'][node_type]:model['end'][node_type]]
                data_list.append(data)
            dataset._data, dataset.slices = dataset.collate(data_list)
        else:
            for node_type in dataset.data.node_types:
                # if hasattr(dataset.data[node_type], 'x'):
                dataset.data[node_type]['pestat_Hetero_Metapath'] = \
                    emb[model['start'][node_type]:model['end'][node_type]]
        print(dataset.data)
    if cfg.posenc_Hetero_TransE.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'TransE'):
            preprocess_KGE(pe_dir, dataset, 'TransE')
        
        model = load_KGE(pe_dir, 'TransE', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_TransE'] = model[node_type].detach().cpu()
    if cfg.posenc_Hetero_ComplEx.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'ComplEx'):
            preprocess_KGE(pe_dir, dataset, 'ComplEx')
        
        model = load_KGE(pe_dir, 'ComplEx', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_ComplEx'] = model[node_type]
    if cfg.posenc_Hetero_DistMult.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'DistMult'):
            preprocess_KGE(pe_dir, dataset, 'DistMult')
        
        model = load_KGE(pe_dir, 'DistMult', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_DistMult'] = model[node_type]
            
    print(dataset[0])

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        if 'LapPE' in pe_enabled_list:
            start = time.perf_counter()
            logging.info(f"Precomputing Positional Encoding statistics: "
                         f"{pe_enabled_list} for all graphs...")
            # Estimate directedness based on 10 graphs to save time.
            is_undirected = all(d.is_undirected() for d in dataset[:10])
            logging.info(f"  ...estimated to be undirected: {is_undirected}")

            pe_dir = osp.join(dataset_dir, 'LapPE')
            file_path = osp.join(pe_dir, f'{dataset.name}.pt')
            if not osp.exists(pe_dir) or not osp.exists(file_path):
                from tqdm import tqdm
                results = []
                for i in tqdm(range(len(dataset)),
                               mininterval=10,
                               miniters=len(dataset)//20):
                    data = compute_posenc_stats(dataset.get(i), 
                                                pe_types=pe_enabled_list,
                                                is_undirected=is_undirected,
                                                cfg=cfg)
                    results.append({'EigVals': data.EigVals, 'EigVecs': data.EigVecs})
                if not osp.exists(pe_dir):
                    os.makedirs(pe_dir)
                torch.save(results, file_path)
            
            from tqdm import tqdm
            results = torch.load(file_path)
            data_list = []
            for i in tqdm(range(len(dataset)),
                        mininterval=10,
                        miniters=len(dataset)//20):
                data = dataset.get(i)
                data.EigVals = results[i]['EigVals']
                data.EigVecs = results[i]['EigVecs']
                data_list.append(data)
            data_list = list(filter(None, data_list))

            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")

        # start = time.perf_counter()
        # logging.info(f"Precomputing Positional Encoding statistics: "
        #              f"{pe_enabled_list} for all graphs...")
        # # Estimate directedness based on 10 graphs to save time.
        # is_undirected = all(d.is_undirected() for d in dataset[:10])
        # logging.info(f"  ...estimated to be undirected: {is_undirected}")
        # pre_transform_in_memory(dataset,
        #                         partial(compute_posenc_stats,
        #                                 pe_types=pe_enabled_list,
        #                                 is_undirected=is_undirected,
        #                                 cfg=cfg),
        #                         show_progress=True
        #                         )
        # elapsed = time.perf_counter() - start
        # timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
        #           + f'{elapsed:.2f}'[-3:]
        # logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # # Precompute in-degree histogram if needed for PNAConv.
    # if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
    #     cfg.gt.pna_degrees = compute_indegree_histogram(
    #         dataset[dataset.data['train_graph_index']])
    #     # print(f"Indegrees: {cfg.gt.pna_degrees}")
    #     # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_DBLP(dataset_dir):
    """Load and preformat DBLP datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = DBLP(root=dataset_dir)
    dataset.data['conference'].x = torch.ones(dataset.data['conference'].num_nodes, 1)
    return dataset


def preformat_IMDB(dataset_dir):
    """Load and preformat IMDB datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = IMDB(root=dataset_dir)
    # data = dataset.data
    # m = data.num_nodes_dict['movie']
    # n = data.num_nodes_dict['director']
    # k = data.num_nodes_dict['movie']
    # edge_index_1 = data[('movie', 'to', 'director')].edge_index#.to(cfg.device)
    # edge_index_2 = data[('director', 'to', 'movie')].edge_index#.to(cfg.device)
    # import torch_sparse
    # edge_index, _ = torch_sparse.spspmm(edge_index_1, torch.ones(edge_index_1.shape[1], device=edge_index_1.device), 
    #                                     edge_index_2, torch.ones(edge_index_2.shape[1], device=edge_index_2.device), 
    #                                     m, n, k, True)
    # data[('movie', 'MDM', 'movie')].edge_index = edge_index

    # m = data.num_nodes_dict['movie']
    # n = data.num_nodes_dict['actor']
    # k = data.num_nodes_dict['movie']
    # edge_index_1 = data[('movie', 'to', 'actor')].edge_index#.to(cfg.device)
    # edge_index_2 = data[('actor', 'to', 'movie')].edge_index#.to(cfg.device)
    # edge_index, _ = torch_sparse.spspmm(edge_index_1, torch.ones(edge_index_1.shape[1], device=edge_index_1.device), 
    #                                     edge_index_2, torch.ones(edge_index_2.shape[1], device=edge_index_2.device), 
    #                                     m, n, k, True)
    # data[('movie', 'MAM', 'movie')].edge_index = edge_index
    return dataset


def preformat_Planetoid(dataset_dir, name):
    """Load and preformat Planetoid datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = Planetoid(root=dataset_dir, name=name, transform=T.NormalizeFeatures(), \
                        split="random", num_train_per_class=3943, num_val=3943, num_test=3943)

    data = HeteroData()
    data['paper'].x = dataset[0].x # assuming there's just one node type.
    data['paper'].y = dataset[0].y.squeeze().masked_fill(torch.isnan(dataset[0].y.squeeze()), -1).type(torch.int64)
    # data['paper', 'cites', 'paper'].edge_index = to_undirected(dataset[0].edge_index) # dataset[0].edge_index # 
    adj_t = get_sparse_tensor(dataset[0].edge_index, num_nodes=dataset[0].x.size(0))
    adj_t = adj_t.to_symmetric()
    data['paper', 'cites', 'paper'].adj_t = adj_t

    split_names = ['train_mask', 'val_mask', 'test_mask']
    for i in range(len(split_names)):
        data['paper'][split_names[i]] = dataset[0][split_names[i]]
        
    dataset.data = data

    return dataset


def preformat_MovieLens(dataset_dir):
    """Load and preformat MovieLens datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = MovieLens(root=dataset_dir)
    return dataset


def preformat_OGB_Node(dataset_dir, name):
    """Load and preformat OGB Node Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    start = time.time()

    # if name == 'ogbn-arxiv':
    #     data = HeteroData()
    #     data['paper'].x = dataset[0].x # assuming there's just one node type.
    #     data['paper'].y = dataset[0].y.squeeze().masked_fill(torch.isnan(dataset[0].y.squeeze()), -1).type(torch.int64)
    #     data['paper'].year = dataset[0].node_year.squeeze()
    #     # data['paper', 'cites', 'paper'].edge_index = to_undirected(dataset[0].edge_index) # dataset[0].edge_index # 
    #     start = time.time()
    #     adj_t = get_sparse_tensor(dataset[0].edge_index, num_nodes=dataset[0].x.size(0))
    #     # end = time.time()
    #     # print("Get sparse tensor cost:", round(end - start, 3), "seconds")
    #     # start = end
    #     adj_t = adj_t.to_symmetric()
    #     data['paper', 'cites', 'paper'].adj_t = adj_t
    #     # end = time.time()
    #     # print("To symmetric cost:", round(end - start, 3), "seconds")
    #     # start = end
    #     # rowptr, col, _ = adj_t.csr()
    #     # end = time.time()
    #     # print("CSR cost:", round(end - start, 3), "seconds")
    #     dataset.data = data

    #     splits = dataset.get_idx_split()
    #     split_names = ['train_mask', 'val_mask', 'test_mask']
    #     for i, key in enumerate(splits.keys()):
    #         mask = index_to_mask(splits[key], size=dataset.data['paper'].y.shape[0])
    #         dataset.data['paper'][split_names[i]] = mask
    #         # set_dataset_attr(dataset, split_names[i], mask, len(mask))
    #     # edge_index = to_undirected(dataset.data.edge_index)
    #     # set_dataset_attr(dataset, 'edge_index', edge_index,
    #     #                  edge_index.shape[1])

    path = osp.join(dataset_dir, '_'.join(name.split('-')), 'transformed')
    if not (osp.exists(path) and osp.isdir(path)) or cfg.dataset.rand_split:
        if name in ['ogbn-mag', 'ogbn-mag-homo-whole', 'mag-year']:
            dataset = PygNodePropPredDataset(name='ogbn-mag', root=dataset_dir)
            graph = dataset[0]
            data = HeteroData()

            # Add edges (sparse adj_t)
            for edge_type in graph.edge_reltype:
                src_type, rel, dst_type = edge_type
                data[(src_type, rel, dst_type)].adj_t = \
                    get_sparse_tensor(graph.edge_index_dict[edge_type], 
                                    num_src_nodes=graph.num_nodes_dict[src_type],
                                    num_dst_nodes=graph.num_nodes_dict[dst_type])
                if src_type == dst_type:
                    data[(src_type, rel, dst_type)].adj_t = \
                        data[(src_type, rel, dst_type)].adj_t.to_symmetric()
                else:
                    row, col = graph.edge_index_dict[edge_type]
                    rev_edge_index = torch.stack([col, row], dim=0)
                    data[(dst_type, 'rev_' + rel, src_type)].adj_t = \
                        get_sparse_tensor(rev_edge_index, 
                                        num_src_nodes=graph.num_nodes_dict[dst_type],
                                        num_dst_nodes=graph.num_nodes_dict[src_type])
            
            # data[('paper', 'PAP', 'paper')].adj_t = (data[('paper', 'rev_writes', 'author')].adj_t.t() @  data[('author', 'writes', 'paper')].adj_t.t()).t()
            # data[('paper', 'PSP', 'paper')].adj_t = (data[('paper', 'has_topic', 'field_of_study')].adj_t.t() @  data[('field_of_study', 'rev_has_topic', 'paper')].adj_t.t()).t()

            # Add node features
            def normalize(mx):
                """Row-normalize sparse matrix"""
                rowsum = np.array(mx.sum(1))
                r_inv = np.power(rowsum, -1).flatten()
                r_inv[np.isinf(r_inv)] = 0.
                r_mat_inv = sp.diags(r_inv)
                mx = r_mat_inv.dot(mx)
                return mx

            data['paper'].x = graph.x_dict['paper'] # torch.cat((graph.x_dict['paper'], torch.log10(deg['paper'].reshape(-1, 1))), axis=-1)
            data['paper'].y = graph.y_dict['paper'].squeeze()

            if not cfg.dataset.rand_split:
                split_idx = dataset.get_idx_split()
                train_paper = split_idx['train']['paper']
                valid_paper = split_idx['valid']['paper']
                test_paper  = split_idx['test']['paper']
            else:
                train_size=.85 # 85%
                valid_size=.6 # 9%
                train_paper, temp_paper = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
                valid_paper, test_paper = train_test_split(temp_paper, train_size=valid_size)
            data['paper'].train_mask = index_to_mask(train_paper, size=graph.y_dict['paper'].shape[0])
            data['paper'].val_mask = index_to_mask(valid_paper, size=graph.y_dict['paper'].shape[0])
            data['paper'].test_mask = index_to_mask(test_paper, size=graph.y_dict['paper'].shape[0])

            # Average paper's embedding to corresponding author and field of study
            for edge_type in [('author', 'writes', 'paper'), ('field_of_study', 'rev_has_topic', 'paper')]:
                src_type, rel, dst_type = edge_type
                edge_index = data.edge_index_dict[edge_type]
                v = torch.ones(edge_index.shape[1])
                m = normalize(sp.coo_matrix((v, edge_index), \
                    shape=(graph.num_nodes_dict[src_type], graph.num_nodes_dict[dst_type])))
                out = m.dot(data[dst_type].x)
                data[src_type].x = torch.from_numpy(out) # torch.cat((torch.from_numpy(out), torch.log10(deg[node_type].reshape(-1, 1))), axis=-1)

            # Average author's embedding to corresponding institution
            edge_type = ('institution', 'rev_affiliated_with', 'author')
            edge_index = data.edge_index_dict[edge_type]
            v = torch.ones(edge_index.shape[1])
            m = normalize(sp.coo_matrix((v, edge_index), \
                shape=(graph.num_nodes_dict['institution'], graph.num_nodes_dict['author'])))
            out = m.dot(data['author'].x)
            data['institution'].x = torch.from_numpy(out) # torch.cat((torch.from_numpy(out), torch.log10(deg['institution'].reshape(-1, 1))), axis=-1)  

            if name == 'ogbn-mag-homo-whole':
                homo = data.to_homogeneous()
                new_data = HeteroData()
                new_data['node'].x = homo.x
                new_data['node'].y = homo.y
                new_data[('node', 'edge', 'node')].edge_index = homo.edge_index

                node_idx = 0
                for idx, node_type in enumerate(data.node_types):
                    if node_type == 'paper':
                        node_idx = idx
                        break
                new_data['node'].train_mask = torch.zeros((homo.num_nodes,), dtype=torch.bool)
                new_data['node'].val_mask = torch.zeros((homo.num_nodes,), dtype=torch.bool)
                new_data['node'].test_mask = torch.zeros((homo.num_nodes,), dtype=torch.bool)
                new_data['node'].train_mask[homo.node_type == node_idx] = data['paper'].train_mask
                new_data['node'].val_mask[homo.node_type == node_idx] = data['paper'].val_mask
                new_data['node'].test_mask[homo.node_type == node_idx] = data['paper'].test_mask
                data = new_data
            elif name == 'mag-year':
                label = even_quantile_labels(graph['node_year']['paper'].squeeze().numpy(), nclasses=5, verbose=False)
                data['paper'].y = torch.from_numpy(label).squeeze() 
                train_size=.5 # 50%
                valid_size=.5 # 25%
                train_idx, temp_idx = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
                val_idx, test_idx = train_test_split(temp_idx, train_size=valid_size)
                data['paper'].train_mask = index_to_mask(train_idx, size=data['paper'].y.shape[0])
                data['paper'].val_mask = index_to_mask(val_idx, size=data['paper'].y.shape[0])
                data['paper'].test_mask = index_to_mask(test_idx, size=data['paper'].y.shape[0])

        elif name == 'ogbn-mag-homo':
            dataset = PygNodePropPredDataset(name='ogbn-mag', root=dataset_dir)
            graph = dataset[0]
            data = HeteroData()

            # Add edges (sparse adj_t)
            edge_type = ('paper', 'cites', 'paper')
            src_type, rel, dst_type = edge_type
            data[(src_type, rel, dst_type)].adj_t = \
                get_sparse_tensor(graph.edge_index_dict[edge_type], 
                                num_src_nodes=graph.num_nodes_dict[src_type],
                                num_dst_nodes=graph.num_nodes_dict[dst_type])
            
            data[(src_type, rel, dst_type)].adj_t = \
                data[(src_type, rel, dst_type)].adj_t.to_symmetric()

            # Add node features
            def normalize(mx):
                """Row-normalize sparse matrix"""
                rowsum = np.array(mx.sum(1))
                r_inv = np.power(rowsum, -1).flatten()
                r_inv[np.isinf(r_inv)] = 0.
                r_mat_inv = sp.diags(r_inv)
                mx = r_mat_inv.dot(mx)
                return mx

            data['paper'].x = graph.x_dict['paper'] # torch.cat((graph.x_dict['paper'], torch.log10(deg['paper'].reshape(-1, 1))), axis=-1)
            data['paper'].y = graph.y_dict['paper']

            split_idx = dataset.get_idx_split()
            train_paper = split_idx['train']['paper']
            valid_paper = split_idx['valid']['paper']
            test_paper  = split_idx['test']['paper']
            data['paper'].train_mask = index_to_mask(train_paper, size=graph.y_dict['paper'].shape[0])
            data['paper'].val_mask = index_to_mask(valid_paper, size=graph.y_dict['paper'].shape[0])
            data['paper'].test_mask = index_to_mask(test_paper, size=graph.y_dict['paper'].shape[0])
        
        elif name == 'ogbn-arxiv' or name == 'ogbn-papers100M':
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
            data = HeteroData()
            data['paper'].x = dataset[0].x # assuming there's just one node type.
            data['paper'].y = dataset[0].y.squeeze().masked_fill(torch.isnan(dataset[0].y.squeeze()), -1).type(torch.int64)
            data['paper'].year = dataset[0].node_year.squeeze()
            data['paper', 'cites', 'paper'].edge_index = to_undirected(dataset[0].edge_index) # dataset[0].edge_index # 
            start = time.time()
            adj_t = get_sparse_tensor(dataset[0].edge_index, num_nodes=dataset[0].x.size(0))
            end = time.time()
            print("Get sparse tensor cost:", round(end - start, 3), "seconds")
            start = end
            adj_t = adj_t.to_symmetric()
            end = time.time()
            print("To symmetric cost:", round(end - start, 3), "seconds")
            start = end
            rowptr, col, _ = adj_t.csr()
            end = time.time()
            print("CSR cost:", round(end - start, 3), "seconds")
            data['paper', 'cites', 'paper'].adj_t = adj_t

            splits = dataset.get_idx_split()
            split_names = ['train_mask', 'val_mask', 'test_mask']
            for i, key in enumerate(splits.keys()):
                mask = index_to_mask(splits[key], size=data['paper'].y.shape[0])
                data['paper'][split_names[i]] = mask
                # set_dataset_attr(dataset, split_names[i], mask, len(mask))
            # edge_index = to_undirected(dataset.data.edge_index)
            # set_dataset_attr(dataset, 'edge_index', edge_index,
            #                  edge_index.shape[1])
        elif name == 'ogbn-products':
            dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
            data = HeteroData()
            data['product'].x = dataset[0].x # assuming there's just one node type.
            # data["product"].node_id = torch.arange(dataset[0].num_nodes)
            data['product'].y = dataset[0].y.squeeze()
            data['product', 'buy_with', 'product'].edge_index = to_undirected(dataset[0].edge_index) # to_undirected(dataset[0].edge_index)

            splits = dataset.get_idx_split()
            split_names = ['train_mask', 'val_mask', 'test_mask']
            for i, key in enumerate(splits.keys()):
                mask = index_to_mask(splits[key], size=data['product'].y.shape[0])
                data['product'][split_names[i]] = mask
        elif name == 'MAG240M':
            dataset = MAG240MDataset(root=dataset_dir)
            num_features = 768
            data = HeteroData()
            
            N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
            x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                        mode='r', shape=(N, num_features))
            
            in_memory = True
            if in_memory:
                temp = np.empty((N, num_features), dtype=np.float16)
                temp[:] = x
                x = torch.from_numpy(temp)

            y = torch.from_numpy(dataset.all_paper_label)
            data['paper'].x = x[:dataset.num_papers]
            data['paper'].y = y.type(torch.long)
            data['author'].x = x[dataset.num_papers : dataset.num_papers+dataset.num_authors]
            data['institution'].x = x[dataset.num_papers+dataset.num_authors :]

            path = f'{dataset.dir}/full_adj_t.pt'
            adj_t_dict = torch.load(path)
            for edge_type, adj_t in adj_t_dict.items():
                data[edge_type].adj_t = adj_t

            train_idx = torch.from_numpy(dataset.get_idx_split('train'))
            val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
            # Test-dev have to submit online. Let's save this for the future
            # test_idx = torch.from_numpy(dataset.get_idx_split('test-dev'))
            val_idx, test_idx = train_test_split(val_idx, train_size=0.5)
            data['paper']['train_mask'] = index_to_mask(train_idx, size=data['paper'].y.shape[0])
            data['paper']['val_mask'] = index_to_mask(val_idx, size=data['paper'].y.shape[0])
            data['paper']['test_mask'] = index_to_mask(test_idx, size=data['paper'].y.shape[0])
        
        if not cfg.dataset.rand_split and name != 'MAG240M' and name != 'ogbn-mag-homo': # Sorry, we can't afford another storage for MAG240M
            os.mkdir(path)
            torch.save(data, osp.join(path, 'data.pt'))
    else:
        data = torch.load(osp.join(path, 'data.pt'))
    
    r'''A fake OGB dataset. 
    Once the OGB dataset is preprocessed and transformed to sparse format, the original OGB
    dataset is no needed any more. We can simply use this fake class to load the preprocessed
    version, rather than load the original dataset and overwrite the data inside.
    Save space and time.
    '''
    dataset = InMemoryDataset(root=dataset_dir)
    dataset.name = name
    dataset.data = data

    end = time.time()
    print('Dataset loading took', round(end - start, 3), 'seconds')
    
    return dataset


def preformat_AML(dataset_dir, name):
    """Load and preformat custom Anti-money Laundering datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific AML dataset

    Returns:
        PyG dataset object
    """
    # transform = T.ToUndirected(merge=True)
    dataset = AMLDataset(root=dataset_dir, name=name, reverse_mp=cfg.dataset.reverse_mp,
                         add_ports=cfg.dataset.add_ports)
    return dataset


def preformat_ETH(dataset_dir):
    """Load and preformat custom Ethereum Phishing Detection datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific AML dataset

    Returns:
        PyG dataset object
    """
    # transform = T.ToUndirected(merge=True)
    dataset = ETHDataset(root=dataset_dir, reverse_mp=cfg.dataset.reverse_mp,
                         add_ports=cfg.dataset.add_ports)
    return dataset



def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


#     @classmethod
#     def from_ogb(self, name: str, root='dataset'):
#         print('Obtaining dataset from ogb...')
#         return self.process_ogb(PygNodePropPredDataset(name=name, root=root))

#     @classmethod
#     def process_ogb(self, dataset):
#         print('Converting to fast dataset format...')
#         data = dataset.data
#         x = to_row_major(data.x).to(torch.float16)
#         y = data.y.squeeze()

#         if y.is_floating_point():
#             y = y.nan_to_num_(-1)
#             y = y.long()

#         adj_t = get_sparse_tensor(data.edge_index, num_nodes=x.size(0))
#         rowptr, col, _ = adj_t.to_symmetric().csr()
#         return self(name=dataset.name, x=x, y=y,
#                    rowptr=rowptr, col=col,
#                    split_idx=dataset.get_idx_split(),
#                    meta_info=dataset.meta_info.to_dict())

#     @classmethod
#     def from_path(self, _path, name):
#         path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
#         if not (path.exists() and path.is_dir()):
#             print(f'First time preprocessing {name}; may take some time...')
#             dataset = self.from_ogb(name, root=_path)
#             print(f'Saving processed data...')
#             dataset.save(_path, name)
#             return dataset
#         else:
#             return self.from_path_if_exists(_path, name)
        
#     def save(self, _path, name):
#         path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
#         # path.mkdir()
#         for i, field in enumerate(self._fields):
#             torch.save(self[i], path.joinpath(field + '.pt'))
    # @classmethod
    # def from_path_if_exists(self, _path, name):
    #     path = Path(_path).joinpath('_'.join(name.split('-')), 'processed')
    #     assert path.exists() and path.is_dir()
    #     data = {
    #         field: torch.load(path.joinpath(field + '.pt'))
    #         for field in self._fields
    #     }
    #     data['y'] = data['y'].long()
    #     data['x'] = data['x'].to(torch.float16)
    #     assert data['name'] == name
    #     return self(**data)