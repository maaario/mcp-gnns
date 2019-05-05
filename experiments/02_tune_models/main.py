import os

from sacred.observers import FileStorageObserver

from clique_finding_models.models.mean_baseline.train_and_evaluate import train_and_evaluate as mean_bl
from clique_finding_models.models.mean_per_degree_baseline.train_and_evaluate import train_and_evaluate as mean_per_degree_bl
from clique_finding_models.models.rank_degree_baseline.train_and_evaluate import train_and_evaluate as rank_deg_bl
from clique_finding_models.models.rank_degree_density_baseline.train_and_evaluate import train_and_evaluate as rank_deg_den_bl
from clique_finding_models.models.chebnet.train_and_evaluate import train_and_evaluate as chebnet
from clique_finding_models.models.structure2vec2.train_and_evaluate import train_and_evaluate as s2v2
from clique_finding_models.tuning import generate_all_configs

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "small_dimacs_bhoslib")
OUTPUTS_DIR = "outputs"
SACRED_DIR = os.path.join(OUTPUTS_DIR, "sacred")

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

if __name__ == "__main__":
    for ex in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl, chebnet, s2v2]:
        ex.observers.append(FileStorageObserver.create(SACRED_DIR))

    for tag, settings in SETTINGS.items():
        # Define paths and set metrics to absolute.
        global_config = dict(
            data_dir=DATA_DIR,
            output_dir=os.path.join(OUTPUTS_DIR, "all"),
            absolute_metrics=True,
            transform_y=settings["transform_y"],
            tag=tag,
        )

        # Train baselines
        for ex in [mean_bl, mean_per_degree_bl, rank_deg_bl, rank_deg_den_bl]:
            ex.run(config_updates=global_config, options={"--loglevel": "WARNING"})
            
        # Train multiple variants (tuning)
        param_config_variants = dict(
            max_epochs = 2,
            batch_size = [8, 16, 32],

            model_hparams = dict(
                embedding_dim = 64,
                conv_layers = [3, 4, 5],
                hidden_dim = 64,
                hidden_layers = [2, 3, 4],
                nonlinearity = "leaky_relu",
                output_type = settings["output"],
            ),
            trainer_hparams = dict(
                learning_rate = [1e-2, 1e-3, 1e-4],
                rank_loss = (tag == "rank"),
            ),
        )
        all_configs = generate_all_configs(param_config_variants)
        for config in all_configs:
            config.update(global_config)
            chebnet.run(config_updates=config, options={"--loglevel": "WARNING"})
            s2v2.run(config_updates=config, options={"--loglevel": "WARNING"})
