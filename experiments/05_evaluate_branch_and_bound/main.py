import os
import subprocess
from tqdm import tqdm

STRATEGIES = [
    "degree",
    "coloring",
    "optimal",
]

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_DIRS = dict()
dir = os.path.join(PROJECT_DIR, "data", "generated", "medium_dimacs_bhoslib")
# dir = os.path.join(PROJECT_DIR, "data", "generated", "large_dimacs_bhoslib")
for subset in os.listdir(dir):
    if subset != "C.100.9":
        DATA_DIRS[subset] = os.path.join(dir, subset)

METRICS_DIR = "outputs/outputs_medium"
# METRICS_DIR = "outputs/outputs_large"
MAXCLIQUE_PATH = os.path.join(PROJECT_DIR, "maxclique", "main_maxclique")

TRAINED_MODEL_DIR = os.path.join(
    "..", "02_tune_models", "selected_trained_models", "chebnet_rel_deg")
TRAINED_MODEL_DIR_2 = os.path.join(
    "..", "02_tune_models", "selected_trained_models", "chebnet_rank")

FNULL = open(os.devnull, "w")


def compute_bb_metrics(branch, bound, metrics_dir, model_dir, data_dir, prefix="", pair=False):
    inputs = sorted([f for f in os.listdir(data_dir)
                     if f.endswith('.in') and f.startswith(prefix)])[:10]

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    for input_file_name in tqdm(inputs):
        input_file_path = os.path.join(data_dir, input_file_name)
        base = os.path.splitext(input_file_name)[0]
        metrics_file_path = os.path.join(metrics_dir, base + ".csv")

        with open(input_file_path, "r") as input_file:
            subprocess.check_call(
                [MAXCLIQUE_PATH, "--branch={}".format(branch), "--bound={}".format(bound),
                    "-m", metrics_file_path, "--branch-dir={}".format(model_dir),
                    "--bound-dir={}".format(model_dir), "--pair-branch" if pair else ""],
                stdin=input_file, stdout=FNULL
            )

for data_set, data_dir in DATA_DIRS.items():
    for bound in ["coloring", "optimal"]:
        for pair in [False]:
            for branch in STRATEGIES:
                metrics_path = os.path.join(METRICS_DIR, data_set, str(pair), branch + "+" + bound)
                compute_bb_metrics(branch, bound, metrics_path, "_", data_dir, pair=pair)        

            metrics_path = os.path.join(METRICS_DIR, data_set, str(pair), "neural:rank+" + bound)
            compute_bb_metrics(
                "neural", bound, metrics_path, TRAINED_MODEL_DIR, data_dir, pair=pair)        

            metrics_path = os.path.join(METRICS_DIR, data_set, str(pair), "neural:rel_deg+" + bound)
            compute_bb_metrics(
                "neural", bound, metrics_path, TRAINED_MODEL_DIR_2, data_dir, pair=pair)

FNULL.close()
