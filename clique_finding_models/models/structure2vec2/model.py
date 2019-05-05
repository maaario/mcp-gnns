import torch
from torch_geometric.nn import MessagePassing

from clique_finding_models.models.model_utils import SaveLoadModelMixin, GraphConvolutionsWithMLP


class S2VConv(MessagePassing):
    def __init__(self, embedding_dim):
        super(S2VConv, self).__init__(aggr="add")
        self.w_1 = torch.nn.Linear(embedding_dim, embedding_dim)
        self.w_2 = torch.nn.Linear(embedding_dim, embedding_dim)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x0):
        return self.w_1(x0) + self.w_2(aggr_out)

    def forward(self, edge_index, x, x0):
        num_nodes = x.size(0)
        return self.propagate(edge_index, x=x, x0=x0, num_nodes=num_nodes)


class Structure2vec2(GraphConvolutionsWithMLP, SaveLoadModelMixin):
    def __init__(self, embedding_dim, conv_layers, hidden_dim, hidden_layers,
                 nonlinearity, output_type, full_initialization):
        super(Structure2vec2, self).__init__(
            embedding_dim, hidden_dim, hidden_layers, nonlinearity, output_type)
        self.full_initialization = full_initialization
        self.max_message_passing_steps = conv_layers
        self.embedding_dim = embedding_dim

        self.conv_layers = torch.nn.ModuleList([S2VConv(embedding_dim)])

    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        
        x0 = torch.zeros((num_nodes, self.embedding_dim), device=x.device)
        if self.full_initialization:
            x0[:, :] = 1
        else:
            x0[:, 0] = 1

        embedding = self.nonlinearF(self.conv_layers[0](edge_index, x0, x0))
        for step in range(self.max_message_passing_steps - 1):
            embedding = self.nonlinearF(self.conv_layers[0](edge_index, embedding, x0))

        return embedding
