from math import floor, log10
import os

from ignite.engine import _prepare_batch, create_supervised_trainer, Events
from ignite.engine.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.handlers import EarlyStopping
from ignite.metrics import Metric, Loss
import torch
from torch.nn import Linear, ModuleList
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import scatter_

from clique_finding_models.metrics import rank_margin_loss


class SaveLoadModelMixin(torch.nn.Module):
    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        path = os.path.join(model_dir, "model_state_dict.pt")
        torch.save(self.state_dict(), path)
        return path

    def load(self, model_dir, map_location=None):
        path = os.path.join(model_dir, "model_state_dict.pt")
        if map_location is None:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
        return path


class GraphConvolutionsWithMLP(torch.nn.Module):
    """
    Base class for simple type of models, where first there are some graph convolution layers
    and on top of that there is a multi-layered perceptron.
    """
    def __init__(self, embedding_dim, hidden_dim, hidden_layers, nonlinearity, output_type):
        super(GraphConvolutionsWithMLP, self).__init__()
        self.embedding_dim = embedding_dim

        self.conv_layers = []

        self.mlp_layers = []
        if hidden_layers == 0:
            self.mlp_layers = [Linear(embedding_dim, 1)]
        else:
            self.mlp_layers = [Linear(embedding_dim, hidden_dim)] + \
                [Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers - 1)] + \
                [Linear(hidden_dim, 1)]
        self.mlp_layers = ModuleList(self.mlp_layers)

        self.nonlinearF = {"relu": F.relu, "tanh": F.tanh, "leaky_relu": F.leaky_relu}[nonlinearity]
        self.outputF = {"linear": lambda x: x, "sigmoid": torch.sigmoid}[output_type]

    def embed(self, data):
        x, edge_index = data.x, data.edge_index

        embedding = self.nonlinearF(self.conv_layers[0](x, edge_index))
        for conv_layer in self.conv_layers[1:]:
            embedding = self.nonlinearF(conv_layer(embedding, edge_index))

        return embedding

    def forward(self, data):
        embedding = self.embed(data)

        for mlp_layer in self.mlp_layers:
            embedding = self.nonlinearF(mlp_layer(embedding))

        return self.outputF(embedding).flatten()


def prepare_batch(batch, device=None, non_blocking=False):
    if device is not None:
        batch = batch.to(device)
    return (batch, batch.y)


def create_supervised_graph_trainer(model, optimizer, loss_fn, device=None, non_blocking=False,
                                    prepare_batch=prepare_batch):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y, x)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_baseline_trainer(model, loss_fn, device=None, non_blocking=False,
                            prepare_batch=prepare_batch):
    """
    Factory function for creating a trainer for supervised models.
    Mostly copied from ignite.engine.create_supervised_trainer, the difference being that no
    gradients are computed nor propagated for baselines.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        return loss.item()

    return Engine(_update)


def param(x, grad=False):
    """
    In order to be able to save baseline parameters, convert other data types to torch.nn.Parameter.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    return torch.nn.Parameter(x, requires_grad=grad)


def create_supervised_graph_evaluator(model, metrics={}, device=None, non_blocking=False,
                                      prepare_batch=prepare_batch):
    """
    Factory function for creating an evaluator for supervised graph models.
    Mostly copied from ignite.engine.create_supervised_evaluator.
    The only change is that the batch vector is passed to metrics, so they can distinguish
    which nodes belong to which graphs.
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return y_pred, y, x

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def rank_loss_transform(x):
    return x[0], x[1], dict(data=x[2])


def prepare_trainer(model, evaluator, device, learning_rate, early_stopping_patience,
                    lr_reduce_patience, lr_reduce_factor, rank_loss):
    """
    Prepare trainer with Adam optimizer, rank margin loss or mse loss, with early stopping
    and ReduceLROnPlateau scheduler.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if rank_loss:
        trainer = create_supervised_graph_trainer(
            model, optimizer, rank_margin_loss, device=device, prepare_batch=prepare_batch)
        Loss(rank_margin_loss, output_transform=rank_loss_transform).attach(evaluator, "loss")
    else:
        trainer = create_supervised_trainer(
            model, optimizer, F.mse_loss, device=device, prepare_batch=prepare_batch)
        Loss(F.mse_loss, output_transform=lambda x: x[:2]).attach(evaluator, "loss")
    trainer.register_events("EVALUATION_COMPLETED")

    # Early stopping
    evaluator.tiny_loss_grow = 0

    def score_function(engine):
        x = evaluator.state.metrics["loss"]
        if x > 0:
            x = round(x, floor(-log10(x * 1e-4)))
        evaluator.tiny_loss_grow += 1e-10
        return -(x + evaluator.tiny_loss_grow)

    handler = EarlyStopping(
        patience=early_stopping_patience, score_function=score_function, trainer=trainer)
    trainer.add_event_handler("EVALUATION_COMPLETED", handler)

    # Decaying learning rate
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_reduce_patience, factor=lr_reduce_factor)

    @trainer.on("EVALUATION_COMPLETED")
    def do_scheduler_step(engine):
        scheduler.step(evaluator.state.metrics["loss"])

    return trainer
