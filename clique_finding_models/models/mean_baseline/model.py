import torch

from clique_finding_models.models.model_utils import SaveLoadModelMixin, param


class MeanBaseline(SaveLoadModelMixin):
    """Predicts a clique size of a vertex as mean clique size from training data."""
    def __init__(self):
        super(MeanBaseline, self).__init__()
        self.num_nodes_processed = param([0])
        self.mean = param([0])

    def forward(self, data):
        if self.training:
            new_sum = data.y.sum() + self.mean * self.num_nodes_processed
            self.num_nodes_processed += len(data.y)
            if self.num_nodes_processed != 0:
                self.mean = param(new_sum / self.num_nodes_processed)

        return torch.ones(data.num_nodes, device=self.mean.device) * self.mean
