import torch
from ogb.utils.features import get_bond_feature_dims

from fraudGT.graphgym.register import register_edge_encoder, register_node_encoder


class ExampleNodeEncoder(torch.nn.Module):
    """
        Provides an encoder for integer node features

        Parameters:
        num_classes - the number of classes for the embedding mapping to learn
    """

    def __init__(self, emb_dim, num_classes=None):
        super(ExampleNodeEncoder, self).__init__()

        self.encoder = torch.nn.Embedding(num_classes, emb_dim)
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.node_feature = self.encoder(batch.node_feature[:, 0])

        return batch


register_node_encoder('example', ExampleNodeEncoder)


class ExampleEdgeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(ExampleEdgeEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        full_bond_feature_dims = get_bond_feature_dims()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_feature.shape[1]):
            bond_embedding += \
                self.bond_embedding_list[i](batch.edge_feature[:, i])

        batch.edge_feature = bond_embedding
        return batch


register_edge_encoder('example', ExampleEdgeEncoder)
