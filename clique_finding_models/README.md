# clique_finding_models 
Infrastructure for training and evaluating clique finding models.

## Setup
```
cd mcp-gnns

# Create virtualenv with Python3.5
python3.5 -m venv env
. env/bin/activate

# Install PyTorch prerequisites
pip install numpy

# Get cuda version
nvcc --version

# Install PyTorch from pytorch.org website, e.g. only for Linux + pip + Python3.5 + no cuda
pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl

# To check if PyTorch was successfully installed, run
python -c "import torch; print(torch.__version__)"

# Install additional requirements
pip install wheel
pip install -r requirements.frozen.txt

# Install this package locally, so that it can be included in experiments 
python setup.py develop
```

## Usage
The main interface provided by this package is [Sacred](https://github.com/IDSIA/sacred)
experiment `train_and_evaluate` to train and evaluate models.

A model can be trained or evaluated either using commmandline 
`python clique_finding_models/models/chebnet/train_and_evaluate.py` or by including 
`train_and_evaluate` instance from this file and then calling `run()` method.

To train and/or evaluate a model, the following arguments can be passed:

* device - `"cuda"` or `"cpu"`, determined automatically
* data_dir - a folder with graph  `.in` and `.out` files (see generator's documentation),
optionally can contain folder structure of subsets, where the leaf nodes are graph in and out files
* train - boolean, whether the model should be trained 
* trained_model_dir - directory from which the model's state dictionary will be loaded - if
no path is supplied, a new instance of a model is created 
* output_dir - path for processed datasets, trained models, predictions and embeddings; use
`train_and_evaluate.observers.append(FileStorageObserver.create(SACRED_DIR))` to store metrics,
metadata and configuration data using Sacred
* evaluate_on_full - if `False`, dataset is split to training and validation in 80:20 ratio, 
otherwise all data is used for validation/testing
* absolute_metrics - if `True`, model outputs are converted to absolute clique sizes
* embeddings - if `True`, vertex embeddings are outputted as PyTorch tensor
* predictions - if `True`, predictions for validaion/testing graphs are outputted as PyTorch tensor
* parameters specifying the training and model architecture: 
max_epochs, batch_size, transform_y, trainer_hparams, model_hparams

Apart from model definitions and `train_and_evaluate`, a few more tools are defined in this package
for model analysis and hyper-parameter tuning.

## Contents of files
* models - model class definitions & customization of `train_and_evaluate` for each model
    * model_utils.py - holds reusable model mixins and trainers
    * model_directory - definitions specific to a single model
        * model.py - model class definition as PyTorch module
        * train_and_evaluate.py - custization of training for a model,
        default values of hyper-parameters
* output_transforms.py - target (clique size) transformations used in training
* graph_dataset.py - dataset loader to load graphs from `.in` files into Pytorch Geometric graphs
* metrics.py - metrics and ranking loss function definition
* experiment.py - reusable implementation of `train_and_evaluate` experiment and methods
  to evaluate a model on separate subsets of a dataset
* tuning.py - grid-search fo hp tuning
* analysis.py - loading Sacred logs and plotting graphs for use in interactive notebooks
* predict.py - functions to load a model and compute its predictions for use in B&B
