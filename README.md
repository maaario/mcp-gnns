# Solving Maximum Clique Problem Using Neural Networks

This repository contains all code necessary to reproduce the experiments described
in my master thesis "Solving Maximum Clique Problem Using Neural Networks" where simpler
Structure2Vec and ChebNet networks are first trained to predict the largest clique size in the
neighbourhood of each vertex or to order vertices according to these values. In the following step,
these networks are used as a branching rule in branch and bound algorithm.

Its main components are:
* `generator` - generator of random graphs to serve as training examples
* `maxclique` - implementation of branch and bound algorithm for labelling example graphs and
evaluating heuristic functions based on neural networks
* `clique_finding_models` - graph neural networks and infrastructure for their training and
evaluation
* `experiments` - scripts and notebooks which produce and visualize the experimental results

For further information about these components read README files in corresponding folders.

## Reproduction
1. Setup `clique_finding_models`, `maxclique`, `generator` following setup sections in their 
README files, preferrably in this order.
2. Generate graphs by running all scripts `python generator/main_*.py`.
3. Run experiments by running scripts `python experiments/*/main.py`.
4. Run `jupyter lab` and visualize experimental results using interactive python notebooks
`experiments/*/*.ipynb` (Restart Kernel and Run All Cells).


### Notes on specific experiments
The experiments `02_tune_models` and `04_structural_generalization` are computation heavy
as they include model training, but we provide best trained model variants in
`02_tune_models/selected_trained_models`. 

Experiment `01_graph_statistics` only requires `small_dimacs_bhoslib` graphs 
to be generated and the statistics can be easily recomputed in jupyter notebook.
Statistics of other datasets can be easily computed with the same notebook when `DATA_DIR`
constant is updated.

Experiment `03_evaluate_models_on_subsets` should be simple to replicate, as only trained models
are evaluated here. Apart from analyzing the performance of networks on graph subsets,
generalization on size can be simply tested by evaluation on larger graphs 
(e.g. as in `subsets_medium_500.ipynb`).
