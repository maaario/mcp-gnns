from abc import abstractmethod

import torch
import torch.nn.functional as F
from ignite.contrib.metrics.average_precision import average_precision_compute_fn
from ignite.metrics import Metric


class AverageGraphMetric(Metric):
    """
    Base class for graph metrics. Calls `compute_for_one_graph` for each graph in batch separately.
    Eventually reports the average of the metric.
    """
    def __init__(self, output_transform=lambda x: x):
        super(AverageGraphMetric, self).__init__(output_transform)
        self._sum_of_values = 0
        self._num_elements = 0

    def reset(self):
        self._sum_of_values = 0
        self._num_elements = 0

    def compute(self):
        if self._num_elements == 0:
            raise NotComputableError(
                "Metric must have at least one example before it can be computed.")
        return self._sum_of_values / self._num_elements

    @abstractmethod
    def compute_for_one_graph(self, y_pred, y):
        pass

    def update(self, output):
        y_pred, y, data = output

        flat_y_pred = y_pred.flatten()
        num_graphs = data.batch[-1].item() + 1

        for graph_id in range(num_graphs):
            self._sum_of_values += self.compute_for_one_graph(
                flat_y_pred[data.batch == graph_id], y[data.batch == graph_id])
            self._num_elements += 1


class PrecisionAtN(AverageGraphMetric):
    """
    Returns 1 for a graph if at least one node of largest clique is in the top N predictions.
    """
    def __init__(self, N, output_transform=lambda x: x):
        super(PrecisionAtN, self).__init__(output_transform)
        self.N = N

    def compute_for_one_graph(self, y_pred, y):
        num_nodes = len(y)
        mc = y.max()

        clique_positions = (y - mc).abs() < 1e-5
        K = min(num_nodes, self.N)
        values, indices = y_pred.topk(k=K)

        hits = clique_positions[indices].sum()
        return hits.item() / K


class FractionOfMaxCliqueNodesFound(AverageGraphMetric):
    """
    Let MC be a set of nodes created as union of all maximum cliques.
    Returns the fraction of nodes from MC in the top MC nodes ranked by predictions.
    """
    def compute_for_one_graph(self, y_pred, y):
        mc = y.max()
        clique_positions = (y - mc).abs() < 1e-5

        mc_size = clique_positions.sum()
        values, indices = y_pred.topk(k=mc_size)

        hits = clique_positions[indices].sum()

        return (hits.item() / mc_size.item())


class AveragePrecision(AverageGraphMetric):
    """
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
    """
    def compute_for_one_graph(self, y_pred, y):
        mc = y.max()
        clique_positions = (y - mc).abs() < 1e-5
        return average_precision_compute_fn(y_pred, clique_positions)


def rank_margin_loss(input, target, data):
    def graph_loss(input, target):
        num_nodes = len(target)

        diffs = F.relu(target * torch.ones(num_nodes, 1, device=data.x.device) - target.view(-1, 1))
        greater = (diffs > 1e-5).float()

        input_rows = input * torch.ones(num_nodes, 1, device=data.x.device)
        pair_losses = F.relu((input_rows.t() - input_rows) * greater + 0.1 * diffs)

        return pair_losses.sum()

    loss = 0
    num_graphs = data.batch[-1].item() + 1
    for graph_id in range(num_graphs):
        loss = loss + graph_loss(input[data.batch == graph_id], target[data.batch == graph_id])
    return loss / num_graphs
