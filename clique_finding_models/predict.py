import os

import jsonpickle
import torch

from clique_finding_models.graph_dataset import load_graph_to_data
from clique_finding_models.models.chebnet.model import ChebNet
from clique_finding_models.models.structure2vec2.model import Structure2vec2
from clique_finding_models.output_transforms import transform_y_dict


config = None
model_cls_from_str = dict(
    s2v2=Structure2vec2,
    chebnet=ChebNet,
)
model = None


def initialize(trained_model_dir):
    global config
    config_path = os.path.join(trained_model_dir, "config.json")
    with open(config_path, "r") as config_file:
        config = jsonpickle.decode(config_file.read())

    global model
    model_cls = model_cls_from_str[config["model"]]
    model = model_cls(**config["model_hparams"])
    model.load(trained_model_dir)


def predict(in_file, out_file):
    transform_y = transform_y_dict[config["transform_y"]]
    graph = transform_y(load_graph_to_data(in_file))

    model.eval()
    with torch.no_grad():
        predictions = model(graph)
        predictions = transform_y.reverse(graph, predictions)

    with open(out_file, "w") as f:
        f.write(" ".join(map(str, predictions.numpy())))
