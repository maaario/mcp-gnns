import torch
from torch.nn import ModuleList
from torch_geometric.nn import ChebConv

from clique_finding_models.models.model_utils import GraphConvolutionsWithMLP, SaveLoadModelMixin


class ChebNet(GraphConvolutionsWithMLP, SaveLoadModelMixin):
    def __init__(self, embedding_dim, conv_layers, polynomial_degree, hidden_dim, hidden_layers,
                 nonlinearity, output_type):
        super(ChebNet, self).__init__(
            embedding_dim, hidden_dim, hidden_layers, nonlinearity, output_type)
        self.num_conv_layers = conv_layers
        self.polynomial_degree = polynomial_degree

        self.conv_layers = ModuleList(
            [ChebConv(1, self.embedding_dim, self.polynomial_degree)] +
            [ChebConv(self.embedding_dim, self.embedding_dim, self.polynomial_degree)
             for _ in range(self.num_conv_layers - 1)])
