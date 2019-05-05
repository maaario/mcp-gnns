import numpy as np
import torch
from torch_geometric.utils import degree

from clique_finding_models.models.model_utils import SaveLoadModelMixin, param


class MeanPerDegreeBaseline(SaveLoadModelMixin):
    """Predicts a clique size of a vertex as mean clique size of vertices with the same degree."""
    def __init__(self, max_degree):
        super(MeanPerDegreeBaseline, self).__init__()
        self.max_degree = max_degree
        self.num_nodes_processed = param(torch.zeros(max_degree + 1))
        self.means = param(torch.zeros(max_degree + 1))

    def forward(self, data):
        degrees = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)

        if self.training:
            for d in range(self.max_degree + 1):
                new_sum = (
                    data.y[degrees == d].sum() +
                    self.means[d] * self.num_nodes_processed[d]
                )
                self.num_nodes_processed[d] += (degrees == d).sum()
                if self.num_nodes_processed[d] != 0:
                    self.means[d] = param(new_sum / self.num_nodes_processed[d])

        return self.means[degrees]
