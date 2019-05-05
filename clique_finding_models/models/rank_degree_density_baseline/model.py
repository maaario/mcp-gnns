import torch
from torch_geometric.utils import degree, sparse_to_dense

from clique_finding_models.models.model_utils import SaveLoadModelMixin


class RankDegreeDensityBaseline(SaveLoadModelMixin):
    """Returns a pair (degree, sum of neighbour degrees) encoded as an integer."""
    def __init__(self):
        super(RankDegreeDensityBaseline, self).__init__()

    def forward(self, data):
        degrees = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
        density = torch.mm(
            degrees.view(1, -1), sparse_to_dense(data.edge_index, num_nodes=data.num_nodes)
        ).flatten()

        return (degrees * 1000000 + density)
