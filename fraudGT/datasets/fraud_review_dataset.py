import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import index_to_mask

from .temporal_dataset import TemporalDataset


class FraudReviewDataset(TemporalDataset):
    """
    Adapter for DGL's FraudAmazonDataset and FraudYelpDataset.

    Supported names:
        'Amazon'  - user review fraud (node classification on 'user' nodes)
        'YelpChi' - restaurant review spam (node classification on 'review' nodes)

    Both datasets have 3 homogeneous relation types but no edge features or
    timestamps. Train/val/test masks are provided by DGL and reused directly.
    """

    CONFIGS = {
        'Amazon': {
            'dgl_cls':   'FraudAmazonDataset',
            'node_type': 'user',
            'edge_types': ['net_upu', 'net_usu', 'net_uvu'],
            'feat_key':  'feature',
        },
        'YelpChi': {
            'dgl_cls':   'FraudYelpDataset',
            'node_type': 'review',
            'edge_types': ['net_rsr', 'net_rtr', 'net_rur'],
            'feat_key':  'feature',
        },
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        assert name in self.CONFIGS, \
            f"name must be one of {list(self.CONFIGS.keys())}, got '{name}'"
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return []  # DGL manages its own download

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        import dgl

        cfg_d = self.CONFIGS[self.name]
        dgl_cls = getattr(dgl.data, cfg_d['dgl_cls'])
        dgl_dataset = dgl_cls(raw_dir=self.raw_dir)
        g = dgl_dataset[0]

        node_type  = cfg_d['node_type']
        edge_types = cfg_d['edge_types']

        x = g.ndata[cfg_d['feat_key']].float()
        y = g.ndata['label'].long()
        num_nodes = g.num_nodes()

        # DGL may store masks as [N, num_splits]; take column 0.
        def _mask(key):
            m = g.ndata[key]
            return m[:, 0].bool() if m.dim() == 2 else m.bool()

        train_mask = _mask('train_mask')
        val_mask   = _mask('val_mask')
        test_mask  = _mask('test_mask')

        # Pre-build edge tensors once (shared across all splits).
        edge_index_dict = {}
        for et in edge_types:
            src, dst = g.edges(etype=et)
            edge_index_dict[et] = torch.stack([src.long(), dst.long()], dim=0)

        for split, split_mask in [('train', train_mask),
                                   ('val',   val_mask),
                                   ('test',  test_mask)]:
            data = HeteroData()
            data[node_type].x          = x
            data[node_type].y          = y
            data[node_type].num_nodes  = num_nodes
            data[node_type].train_mask = train_mask
            data[node_type].val_mask   = val_mask
            data[node_type].test_mask  = test_mask
            data[node_type].split_mask = split_mask

            for et in edge_types:
                data[node_type, et, node_type].edge_index = edge_index_dict[et]
                # No edge features; model detects this via num_edge_features == 0

            self.data_dict[split] = data

        torch.save(self.data_dict, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'FraudReviewDataset(name={self.name})'
