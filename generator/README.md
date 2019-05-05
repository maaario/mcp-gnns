# Random graph generator 

## Set up
1. Install requirements for the whole project
2. Build the branch and bound (`maxclique`)
3. Download generators
```
cd generator
./download_third_party_generators.sh
```

## Usage
To generate graphs from iteration:
```
python generator/main_<generation iteration>.py 
```
Graphs will be generated to paths specified in these files, preferrably in `/data/generated`
folder. 

## Generated files .in, .out, .json
For each generated graph, 3 files are created:
```
/data/generated/<generation iteration>/<graph type>/00000.in, 00000.out, 00000.in.json
```

All graphs are generated in text format `.in` with number od nodes and edges in the first line
and pairs of adjacent vertices in other lines.
```
<num nodes> <num edges>
<node 1> <node 2>
<node 1> <node 2>
...
```

File `.json` holds the parameters with which the graph was generated. 

File `.out` holds the clique sizes for each node - the size of maximum clique in graph induced
by node v and its neighbours.

## Extending generators
All reusable generator code is in `gen_methods.py` file.

New generator method should be named with prefix `generate_`, has to take `path` and `seed`
as 2 first positional arguments and it should produce the `.in` file to `path` and
store generation parameters to `path` using `store_parameters(path, **kwargs)` function. 

With this convention, multiple graphs of this type can be easily generated 
using `generate_multiple_graphs(output_dir, num_graphs, method, seed=0, labels=True, **kwargs)`.
This function repeatedly calls the provided generating `method`, automatically generates random 
seeds for graphs based on `seed` and if `labels` is set, it also finds cliques in these 
graphs using maxclique tool.  
