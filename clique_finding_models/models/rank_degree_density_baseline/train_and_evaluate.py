import torch
import torch.nn.functional as F

from clique_finding_models.experiment import set_up_experiment
from clique_finding_models.models.rank_degree_density_baseline.model import RankDegreeDensityBaseline
from clique_finding_models.models.model_utils import prepare_batch, create_baseline_trainer


def prepare_trainer(model, evaluator, device):
    trainer = create_baseline_trainer(
        model, F.mse_loss, device=device, prepare_batch=prepare_batch)
    return trainer

train_and_evaluate = set_up_experiment(RankDegreeDensityBaseline, prepare_trainer)


@train_and_evaluate.config
def hparams():
    max_epochs = 1
    model = "rank_deg_den_bl"

if __name__ == '__main__':
    train_and_evaluate.run_commandline()
