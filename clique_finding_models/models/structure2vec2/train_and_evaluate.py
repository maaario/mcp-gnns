from clique_finding_models.experiment import set_up_experiment
from clique_finding_models.models.model_utils import prepare_trainer
from clique_finding_models.models.structure2vec2.model import Structure2vec2

train_and_evaluate = set_up_experiment(Structure2vec2, prepare_trainer)


@train_and_evaluate.config
def hparams():
    model = "s2v2"
    max_epochs = 200
    trainer_hparams = dict(
        learning_rate=1e-4,
        early_stopping_patience=10,
        lr_reduce_patience=5,
        lr_reduce_factor=0.2,
        rank_loss = False,
    )
    model_hparams = dict(
        embedding_dim=64,
        conv_layers=4,
        hidden_dim=64,
        hidden_layers=2,
        output_type="sigmoid",
        nonlinearity="relu",
        full_initialization=False,
    )

if __name__ == '__main__':
    train_and_evaluate.run_commandline()
