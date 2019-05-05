import os

from sacred.observers import FileStorageObserver

from clique_finding_models.experiment import last_eid, evaluate_model_on_subsets
from clique_finding_models.models.mean_baseline.train_and_evaluate import train_and_evaluate as mean_bl
from clique_finding_models.models.mean_per_degree_baseline.train_and_evaluate import train_and_evaluate as mean_per_degree_bl
from clique_finding_models.models.rank_degree_baseline.train_and_evaluate import train_and_evaluate as rank_deg_bl
from clique_finding_models.models.rank_degree_density_baseline.train_and_evaluate import train_and_evaluate as rank_deg_den_bl
from clique_finding_models.models.structure2vec2.train_and_evaluate import train_and_evaluate as s2v2
from clique_finding_models.models.chebnet.train_and_evaluate import train_and_evaluate as chebnet

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# To produce the results of size-generalization experiments, uncomment the following 2 lines
# and comment next 2 lines.
# DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "medium_dimacs_bhoslib_500")
# OUTPUTS_DIR = "outputs/medium_500"

DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "small_dimacs_bhoslib_500")
OUTPUTS_DIR = "outputs/small_500"
SACRED_DIR = os.path.join(OUTPUTS_DIR, "sacred")

TRAINED_MODELS_DIR = os.path.join("..", "02_tune_models", "selected_trained_models")

SETTINGS = {
    "rel_mc": {
        "transform_y": "relative_to_max_clique",
        "output": "sigmoid",
    },
    "rel_deg": {
        "transform_y": "relative_to_degree",
        "output": "sigmoid",
    },
    "abs": {
        "transform_y": "none",
        "output": "linear",
    },
    "rank": {
        "transform_y": "none",
        "output": "linear",
    }
}

for ex in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl, s2v2, chebnet]:
    ex.observers.append(FileStorageObserver.create(SACRED_DIR))

# Train and evaluate baselines
for tag, settings in SETTINGS.items():
    global_config = dict(
        data_dir=DATA_DIR,
        output_dir=os.path.join(OUTPUTS_DIR, "all"),
        absolute_metrics=True,
        transform_y=settings["transform_y"],
        tag=tag,
    )
    for ex in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl]:
        ex.run(config_updates=global_config, options={"--loglevel": "WARNING"})
        model_dir = os.path.join(SACRED_DIR, str(last_eid(SACRED_DIR)))
        evaluate_model_on_subsets(DATA_DIR, OUTPUTS_DIR, model_dir, ex, evaluate_all=True)

    for subset in os.listdir(DATA_DIR):
        global_config["data_dir"] = os.path.join(DATA_DIR, subset)
        global_config["output_dir"] = os.path.join(OUTPUTS_DIR, subset)
        for ex in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl]:
            ex.run(config_updates=global_config, options={"--loglevel": "WARNING"})

# Evaluate best trained models
for model_dir in os.listdir(TRAINED_MODELS_DIR):
    model_full_dir = os.path.join(TRAINED_MODELS_DIR, model_dir)
    model = chebnet if model_dir.startswith("chebnet") else s2v2
    evaluate_model_on_subsets(DATA_DIR, OUTPUTS_DIR, model_full_dir, model, evaluate_all=True)
