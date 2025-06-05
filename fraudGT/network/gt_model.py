import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from fraudGT.graphgym.models import head  # noqa, register module
from fraudGT.graphgym import register as register
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_network
from fraudGT.graphgym.models.layer import BatchNorm1dNode
from torch_geometric.utils import (to_undirected, to_dense_batch)

from fraudGT.layer.gt_layer import GTLayer
from fraudGT.network.utils import GTPreNN
from torch_geometric.nn import (Sequential, Linear, HeteroConv, GraphConv, SAGEConv, HGTConv, GATConv)

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dataset):
        super(FeatureEncoder, self).__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gt.dim_hidden, dataset)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.gt.dim_hidden)
            # Update dim_in to reflect the new dimension of the node features
            if self.is_hetero:
                self.dim_in = {node_type: cfg.gt.dim_hidden for node_type in dim_in}
            else:
                self.dim_in = cfg.gt.dim_hidden
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gt.dim_hidden)
            else:
                cfg.gnn.dim_edge = cfg.gt.dim_hidden
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge, dataset)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GTModel')
class GTModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero  = isinstance(dataset[0], HeteroData)
        if self.is_hetero:
            self.metadata   = dataset[0].metadata()
        else:
            self.metadata = [("node_type",), (("node_type", "edge_type", "node_type"), )]
        self.dim_h      = cfg.gt.dim_hidden
        self.input_drop = nn.Dropout(cfg.gt.input_dropout)
        self.activation = register.act_dict[cfg.gt.act]
        self.batch_norm = cfg.gt.batch_norm
        self.layer_norm = cfg.gt.layer_norm
        self.l2_norm    = cfg.gt.l2_norm
        GNNHead         = register.head_dict[cfg.gt.head]

        self.encoder = FeatureEncoder(dim_in, dataset)
        self.dim_in = self.encoder.dim_in


        if cfg.gt.layers_pre_gt > 0:
            if not self.is_hetero:
                self.dim_in = {self.metadata[0][0]: self.dim_in}
            self.pre_gt_dict = torch.nn.ModuleDict()
            for node_type in self.metadata[0]:
                self.pre_gt_dict[node_type] = GTPreNN(
                    self.dim_in[node_type], self.dim_h,
                    has_bn=self.batch_norm, has_ln=self.layer_norm,
                    has_l2norm=self.l2_norm
                )
        
        
        try:
            layer_type = cfg.gt.layer_type
            if layer_type in ['TorchTransformer', 'SparseNodeTransformer']:
                local_gnn_type, global_model_type = 'None', layer_type
            else:
                local_gnn_type, global_model_type = layer_type, 'None'
        except:
            raise ValueError(f"Unexpected layer type: {layer_type}")

        self.num_virtual_nodes = cfg.gt.virtual_nodes
        if self.num_virtual_nodes > 0:
            self.virtual_nodes = nn.ParameterDict()
            for node_type in dim_in:
                self.virtual_nodes[node_type] = nn.Parameter(torch.empty((self.num_virtual_nodes, self.dim_h)))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim_h_total = self.dim_h
        for i in range(cfg.gt.layers):
            conv = GTLayer(self.dim_h, self.dim_h, self.dim_h, self.metadata,
                    local_gnn_type, global_model_type, i,
                    cfg.gt.attn_heads, 
                    # layer_norm=False,
                    # batch_norm=False)
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    return_attention=False)
            self.convs.append(conv)

            if self.layer_norm or self.batch_norm:
                self.norms.append(nn.ModuleDict())
                for node_type in self.metadata[0]:
                    if self.layer_norm:
                        self.norms[-1][node_type] = nn.LayerNorm(self.dim_h)
                    elif self.batch_norm:
                        self.norms[-1][node_type] = nn.BatchNorm1d(self.dim_h)
            
            if cfg.gt.residual == 'Concat':
                self.dim_h *= 2
            if cfg.gt.jumping_knowledge:
                dim_h_total += self.dim_h
            else:
                dim_h_total = self.dim_h

        self.post_gt = GNNHead(dim_h_total, dim_out, dataset)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_virtual_nodes > 0:
            for node_type in self.virtual_nodes:
                torch.nn.init.normal_(self.virtual_nodes[node_type])

    def forward(self, batch, return_embedding: bool = False):
        batch = self.encoder(batch)
        if isinstance(batch, HeteroData):
            h_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
        else:
            h_dict = {self.metadata[0][0]: batch.x}
            edge_index_dict = {self.metadata[1][0]: batch.edge_index}

        h_dict = {
            node_type: self.input_drop(h_dict[node_type]) for node_type in h_dict
        }

        if cfg.gt.layers_pre_gt > 0:
            h_dict = {
                node_type: self.pre_gt_dict[node_type](h_dict[node_type]) for node_type in h_dict
            }

        interm = {node_type: [h_dict[node_type]] for node_type in h_dict}
        num_nodes_dict = None
        if self.num_virtual_nodes > 0:
            # Concat global virtual nodes to the end, so the edge_index leaves untouched
            h_dict = {
                node_type: torch.cat((h, self.virtual_nodes[node_type]), dim=0)
                for node_type, h in h_dict.items()
            }
            # Connect global virtual nodes to every node
            num_nodes_dict = batch.num_nodes_dict
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                rows, cols = [], []
                for i in range(self.num_virtual_nodes):
                    rows.append(torch.full((1, num_nodes_dict[dst_type]), num_nodes_dict[src_type] + i, device=edge_index_dict[edge_type].device))
                    cols.append(torch.arange(num_nodes_dict[dst_type], device=edge_index_dict[edge_type].device).view(1, -1))

                edge_index_dict[edge_type] = torch.cat((edge_index_dict[edge_type], torch.cat((torch.cat(rows, dim=-1), torch.cat(cols, dim=-1)))), dim=-1)
                if src_type == dst_type:
                    edge_index_dict[edge_type] = torch.cat((edge_index_dict[edge_type], torch.cat((torch.cat(cols, dim=-1), torch.cat(rows, dim=-1)))), dim=-1)

            for node_type in batch.node_types:
                batch[node_type].num_nodes += self.num_virtual_nodes

        # Write back for conv layer
        if isinstance(batch, HeteroData):
            for node_type in batch.node_types:
                batch[node_type].x = h_dict[node_type]
        else:
            batch.x = h_dict[self.metadata[0][0]]
        for i in range(cfg.gt.layers):
            batch = self.convs[i](batch)
            # batch = self.convs[i](batch)
            # batch.saved_scores = saved_scores
            if cfg.gt.jumping_knowledge:
                h_temp_dict = batch.collect('x')
                if self.num_virtual_nodes > 0:
                    # Remove the virtual nodes
                    h_temp_dict = {
                        node_type: h[:num_nodes_dict[node_type], :] for node_type, h in h_dict.items()
                    }
                for node_type in h_dict:
                    interm[node_type] = interm[node_type] + [h_temp_dict[node_type]]

        if self.num_virtual_nodes > 0:
            # Remove the virtual nodes
            for node_type in batch.node_types:
                batch[node_type].x = batch[node_type].x[:num_nodes_dict[node_type], :]
                batch[node_type].num_nodes -= self.num_virtual_nodes

        # Jumping knowledge.
        if cfg.gt.jumping_knowledge:
            for node_type in batch.node_types:
                batch[node_type].x = torch.cat(interm[node_type], dim=1)

        # Output L2 norm
        if cfg.gt.l2_norm:
            for node_type in batch.node_types:
                batch[node_type].x = F.normalize(batch[node_type].x, p=2, dim=-1) 

        return self.post_gt(batch, return_embedding)
