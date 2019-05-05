import torch
from torch_geometric.utils import degree

from clique_finding_models.models.model_utils import SaveLoadModelMixin


class RankDegreeBaseline(SaveLoadModelMixin):
    """Returns the vertex degree."""
    def __init__(self):
        super(RankDegreeBaseline, self).__init__()

    def forward(self, data):
        degrees = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float)
        return degrees
