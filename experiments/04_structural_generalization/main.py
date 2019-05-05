import jsonpickle
import os

from sacred.observers import FileStorageObserver

from clique_finding_models.experiment import last_eid, evaluate_model_on_subsets
from clique_finding_models.models.mean_baseline.train_and_evaluate import train_and_evaluate as mean_bl
from clique_finding_models.models.mean_per_degree_baseline.train_and_evaluate import train_and_evaluate as mean_per_degree_bl
from clique_finding_models.models.chebnet.train_and_evaluate import train_and_evaluate as chebnet
from clique_finding_models.models.rank_degree_baseline.train_and_evaluate import train_and_evaluate as rank_deg_bl
from clique_finding_models.models.rank_degree_density_baseline.train_and_evaluate import train_and_evaluate as rank_deg_den_bl
from clique_finding_models.models.structure2vec2.train_and_evaluate import train_and_evaluate as s2v2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "small_dimacs_bhoslib_500")
OUTPUTS_DIR = "outputs"
SACRED_DIR = os.path.join(OUTPUTS_DIR, "sacred")
for ex in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl, chebnet, s2v2]:
    ex.observers.append(FileStorageObserver.create(SACRED_DIR))
TRAINED_MODELS_DIR = os.path.join("..", "02_tune_models", "selected_trained_models")

def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r") as config_file:
        config = jsonpickle.decode(config_file.read())
        config = dict(model_hparams=config["model_hparams"],
                      trainer_hparams=config["trainer_hparams"])
    return config

max_epochs = 200

variants = [
    {   # rel_deg
        "config": load_config(os.path.join(TRAINED_MODELS_DIR, "chebnet_rel_deg")),
        "transform_y": "relative_to_degree",
        "max_epochs": max_epochs,
        "model": chebnet,
    },
    {   # rank
        "config": load_config(os.path.join(TRAINED_MODELS_DIR, "chebnet_rank")),
        "transform_y": "none",
        "max_epochs": max_epochs,
        "model": chebnet,
    },
    {   # abs
        "config": load_config(os.path.join(TRAINED_MODELS_DIR, "s2v_abs")),
        "transform_y": "none",
        "max_epochs": max_epochs,
        "model": s2v2,
    },
]

for baseline in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl]:
    variants.append({
        "config": dict(),
        "transform_y": "none",
        "max_epochs": 1,
        "model": baseline,
    })

subsets = os.listdir(DATA_DIR)
for variant in variants:
    for training_set in subsets + [""]:
        config = dict(
            data_dir = os.path.join(DATA_DIR, training_set),
            output_dir = os.path.join(OUTPUTS_DIR, training_set if training_set else "all"),
            absolute_metrics=True,
            tag=training_set if training_set else "all",
            transform_y=variant["transform_y"],
            max_epochs=variant["max_epochs"],
        )
        config.update(variant["config"])
        variant["model"].run(config_updates=config, options={"--loglevel": "WARNING"})
        evaluate_model_on_subsets(
            DATA_DIR, OUTPUTS_DIR, os.path.join(SACRED_DIR, str(last_eid(SACRED_DIR))),
            variant["model"], evaluate_all=True,
        )
