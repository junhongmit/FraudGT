import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


@register_head('hetero_node')
class HeteroGNNNodeHead(nn.Module):
    r'''Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    '''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
                                 bias=True)

    def _apply_index(self, batch, return_embedding: bool = False):
        task = cfg.dataset.task_entity
        # The front [:batch_size] nodes are the original input nodes in HGTLoader
        if isinstance(batch, HeteroData):
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return batch[task].x[:batch_size], \
                    batch[task].y[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return batch[task].x[batch[task][mask]], \
                    batch[task].y[batch[task][mask]]
        else:
            mask = f'{batch.split}_mask'
            return batch.x[batch[mask]], batch.y[batch[mask]]
        
    def _apply_index_with_embedding(self, batch, ori_x):
        task = cfg.dataset.task_entity
        # The front [:batch_size] nodes are the original input nodes in HGTLoader
        if isinstance(batch, HeteroData):
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return batch[task].x[:batch_size], \
                    batch[task].y[:batch_size], \
                    ori_x[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return batch[task].x[batch[task][mask]], \
                    batch[task].y[batch[task][mask]], \
                    ori_x[batch[task][mask]]
        else:
            mask = f'{batch.split}_mask'
            return batch.x[batch[mask]], batch.y[batch[mask]], ori_x[batch[mask]]

    def forward(self, batch, return_embedding: bool = False):
        if isinstance(batch, HeteroData):
            ori_x = x = batch[cfg.dataset.task_entity].x
            x = self.layer_post_mp(x)
            batch[cfg.dataset.task_entity].x = x
        else:
            ori_x = batch.x
            batch.x = self.layer_post_mp(batch.x)

        if not return_embedding:
            pred, label = self._apply_index(batch)
            return pred, label
        else:
            pred, label, ori_x = self._apply_index_with_embedding(batch, ori_x)
            return pred, label, ori_x
