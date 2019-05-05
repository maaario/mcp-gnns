import os

import gen_methods

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "large_dimacs_bhoslib")

num_graphs = 100

if __name__ == "__main__":
    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "C.100.9"), num_graphs=num_graphs, seed=13247,
        method=gen_methods.generate_gnp_graph, num_vertices=100, probability=0.9)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "C.100.5"), num_graphs=num_graphs, seed=13248,
        method=gen_methods.generate_gnp_graph, num_vertices=100, probability=0.5)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "dsjc100"), num_graphs=num_graphs, seed=13249,
        method=gen_methods.generate_dsjc_random_k, num_vertices=100)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "brock100"), num_graphs=num_graphs, seed=13259,
        method=gen_methods.generate_brockington_random, num_vertices=100)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "rb13-8"), num_graphs=num_graphs, seed=13252,
        method=gen_methods.generate_rb_maxclique, n=13, a=0.8, p=0.25, r=2.7808)
