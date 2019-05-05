# maxclique
A tool for 
* finding maximum clique in graph using branch and bound,
* creating training labels - maximum clique size for each vertex
* testing graph neural networks as heuristic functions of B&B
* storing metrics about runs of B&B

## Setup
```
cd maxclique

# Download cxxopts header
make install

# Compile
make
```

## Usage
```
$ ./main_maxclique --help
Find the maximum clique with branch and bound.
Usage:
  ./main_maxclique [OPTION...]

      --branch arg      branching strategy (default: coloring)
      --bound arg       bounding strategy (default: coloring)
  -m, --metrics arg     metrics file path (default: )
      --branch-dir arg  directory of a trained neural model to use for
                        branching (default: )
      --bound-dir arg   directory of a trained neural model to use for
                        bounding (default: )
      --pair-branch     for branch score use pair (bound, branch)
      --max-calls arg   kill after 'max-calls' recursive calls (default: 0)
      --help            print help
```

The input file in `.in` format is read through the standard input and the size of the
maximum clique and the ids of its vertices are printed to standard output.
Use `./main_maxclique <input.in >output.out` to use files.

Labels for graphs can be computed using another main, `./main_clique_for_each_vertex`.
No arguments are passed, again standard IO is used.

To evaluate neural models, the `main_maxlique` has to be run in an environmnet where 
`clique_finding_models` package is installed, e.g. in virtualenv.

## Testing
Run `make test` to run B&B on small graphs.

## Project architecture
* graph - manages a graph structure as a list of neighbours for each vertex
* branch_bound_strategy - strategies to compute branching and bounding scores in each call of B&B
* metrics - metrics manager capturing metrics in B&B calls and outputting them to file
* settings - an object holding all settings parsed from commandline arguments, defaults for
`main_clique_for_each_vertex` are defined here, while defaults for `main_maxclique`
are directly defined in the main file
* maxclique_utils - the implementation of branch and bound algorithm

### Extending B&B strategies 
A new strategy for branch or bound can be easily added by implementing a single function
`vector<Candidate> compute_scores(const Graph& graph)` in a new subclass of an abstract class 
`Strategy`. Finally, to enable the usage of this strategy via commandline arguments, add a 
record to `strategy_map` in constructor of the strategy manager `BranchBoundStrategy` in `branch_and_bound_strategy.cpp`.

### Communication with PyTorch models
Currently, the communication with neural models is implemented via file system and python shell
in `NeuralStrategy`. The subgraph induced by candidate vertices is first written into
`temp_input_graph.in` file, then function `predict.predict` from `clique_finding_models` folder
is called and finally, the predictions (heuristic scores) are read from `temp_clique_sizes.out`
file.
