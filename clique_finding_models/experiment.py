from collections import OrderedDict
import jsonpickle
import os

from ignite.engine import Events
from ignite.metrics import Loss
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm

from clique_finding_models.graph_dataset import GraphDataset
from clique_finding_models.metrics import PrecisionAtN, FractionOfMaxCliqueNodesFound, AveragePrecision
from clique_finding_models.models.model_utils import create_supervised_graph_evaluator, prepare_batch
from clique_finding_models.output_transforms import transform_y_dict


def run_and_log_training(trainer, evaluator, train_loader, test_loader, max_epochs, _run):
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    trainer.register_events("EVALUATION_COMPLETED")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update()

    @trainer.on(Events.STARTED)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        pbar.n = pbar.last_print_n = 0
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        metrics_str = ", ".join(["{}: {:.3f}".format(k, v) for k, v in metrics.items()])
        pbar.write("\rTraining Results - Epoch: {} Metrics: {}"
                   .format(engine.state.epoch, metrics_str))
        for metric_name, metric_value in metrics.items():
            _run.log_scalar("train." + metric_name, metric_value, engine.state.epoch)

    @trainer.on(Events.STARTED)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        metrics_str = ", ".join(["{}: {:.3f}".format(k, v) for k, v in metrics.items()])
        pbar.write("\rValidation Results - Epoch: {} Metrics: {}"
                   .format(engine.state.epoch, metrics_str))
        for metric_name, metric_value in metrics.items():
            _run.log_scalar("val." + metric_name, metric_value, engine.state.epoch)
        engine.fire_event("EVALUATION_COMPLETED")

    trainer.run(train_loader, max_epochs=max_epochs)
    pbar.close()


def compute_embeddings(model, data_dir, output_dir, device):
    root = os.path.join(output_dir, "data")
    data = GraphDataset(root=root, source_data_dir=data_dir, full=True)

    embeddings = list()
    for graph in data:
        graph = graph.to(device)
        embeddings.append(model.embed(graph))

    embeddings = torch.cat(embeddings, 0)

    path = os.path.join(output_dir, "embeddings.pt")
    torch.save(embeddings, path)
    return path


def compute_predictions(model, data_loader, transform_y, output_dir, device):
    predictions = list()

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x, y = prepare_batch(batch, device=device)
            y_pred = model(x)
            y_pred = transform_y.reverse(x, y_pred)
            predictions.append(y_pred)

    predictions = torch.cat(predictions)

    path = os.path.join(output_dir, "predictions.pt")
    torch.save(predictions, path)
    return path


def set_up_experiment(model_cls, prepare_trainer):
    ex = Experiment('train and evaluate')
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.config
    def config():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_dir = None
        trained_model_dir = None
        output_dir = None

        train = True
        max_epochs = 100
        batch_size = 32
        transform_y = "relative_to_max_clique"

        model = None
        trainer_hparams = dict()
        model_hparams = dict()

        evaluate_on_full = False
        absolute_metrics = False

        embeddings = False
        predictions = False

        tag = None

    @ex.main
    def train_and_evaluate(device, data_dir, trained_model_dir, output_dir, train, max_epochs,
                           batch_size, transform_y, trainer_hparams, model_hparams,
                           evaluate_on_full, absolute_metrics, embeddings, predictions,
                           _run, _seed):
        torch.manual_seed(_seed)
        # Prepare train (optional) and test data loaders.
        transform_y = transform_y_dict[transform_y]
        root = os.path.join(output_dir, "data")
        if train:
            train_data = GraphDataset(root=root, source_data_dir=data_dir, train=True,
                                      transform=transform_y)
            ex.add_resource(train_data.processed_data_path)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_data = GraphDataset(root=root, source_data_dir=data_dir, train=False,
                                 full=evaluate_on_full, transform=transform_y)
        ex.add_resource(test_data.processed_data_path)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Set up model and evaluator.
        model = model_cls(**model_hparams).to(device)
        if trained_model_dir is not None:
            path = model.load(trained_model_dir)
            ex.add_resource(path)

        def output_transform(inference_output):
            y_pred, y, data = inference_output
            if absolute_metrics:
                y_pred, y = transform_y.reverse(data, y_pred), transform_y.reverse(data, y)
            return y_pred, y, data

        evaluator = create_supervised_graph_evaluator(
            model, metrics=OrderedDict(
                mse = Loss(F.mse_loss, output_transform=lambda x: output_transform(x)[:2]),
                mae = Loss(F.l1_loss, output_transform=lambda x: output_transform(x)[:2]),
                top1 = PrecisionAtN(1, output_transform=output_transform),
                top5 = PrecisionAtN(5, output_transform=output_transform),
                frac_mc = FractionOfMaxCliqueNodesFound(output_transform=output_transform),
                avep = AveragePrecision(output_transform=output_transform),
            ),
            device=device,
            prepare_batch=prepare_batch,
        )

        # Train (optional) and evaluate.
        if train:
            trainer = prepare_trainer(model, evaluator, device, **trainer_hparams)
            run_and_log_training(
                trainer, evaluator, train_loader, test_loader, max_epochs, _run)
            new_trained_model_dir = os.path.join(output_dir, "model")
            path = model.save(new_trained_model_dir)
            ex.add_artifact(path)
        else:
            evaluator.run(test_loader)
            for name, value in evaluator.state.metrics.items():
                _run.log_scalar("val." + name, value, 0)

        evaluator.run(test_loader)
        for name, value in evaluator.state.metrics.items():
            print("{}: {:.3f}".format(name, value))

        if embeddings:
            path = compute_embeddings(model, data_dir, output_dir, device)
            ex.add_artifact(path)

        if predictions:
            path = compute_predictions(model, test_loader, transform_y, output_dir, device)
            ex.add_artifact(path)

    return ex


def last_eid(sacred_dir):
    return max([int(x) for x in os.listdir(sacred_dir) if not x.startswith("_")])


def evaluate_model_on_subsets(data_dir, processed_data_dir, trained_model_dir, train_and_evaluate,
                              evaluate_all=False, evaluate_on_full=False):
    """
    Loads a trained model from trained_model_dir, evaluates it on subsets of data.
    train_and_evaluate experiment instance must correspond with stored trained model.
    """
    config_path = os.path.join(trained_model_dir, "config.json")
    with open(config_path, "r") as config_file:
        config = jsonpickle.decode(config_file.read())

    for subset in os.listdir(data_dir):
        config.update(dict(
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            data_dir = os.path.join(data_dir, subset),
            output_dir = os.path.join(processed_data_dir, subset),
            train = False,
            trained_model_dir = trained_model_dir,
            evaluate_on_full = evaluate_on_full,
        ))
        train_and_evaluate.run(config_updates=config, options={"--loglevel": "WARNING"})

    if evaluate_all:
        config.update(dict(
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            data_dir = data_dir,
            output_dir = os.path.join(processed_data_dir, "all"),
            train = False,
            trained_model_dir = trained_model_dir,
            evaluate_on_full = evaluate_on_full,
        ))
        train_and_evaluate.run(config_updates=config, options={"--loglevel": "WARNING"})


def evaluate_on_subsets(data_dir, processed_data_dir, sacred_dir, eid, train_and_evaluate):
    """For backwards compatibility with older experiments."""
    trained_model_dir = os.path.join(sacred_dir, str(eid))
    evaluate_model_on_subsets(data_dir, processed_data_dir, trained_model_dir, train_and_evaluate)
