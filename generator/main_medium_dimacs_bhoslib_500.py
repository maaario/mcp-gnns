import os

import gen_methods

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "medium_dimacs_bhoslib_500")

num_graphs = 500

if __name__ == "__main__":
    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "C.50.9"), num_graphs=num_graphs, seed=247,
        method=gen_methods.generate_gnp_graph, num_vertices=50, probability=0.9)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "C.50.5"), num_graphs=num_graphs, seed=248,
        method=gen_methods.generate_gnp_graph, num_vertices=50, probability=0.5)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "dsjc50"), num_graphs=num_graphs, seed=249,
        method=gen_methods.generate_dsjc_random_k, num_vertices=50)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "brock50"), num_graphs=num_graphs, seed=267,
        method=gen_methods.generate_brockington_random, num_vertices=50, max_tries=20)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "rb9-6"), num_graphs=num_graphs, seed=252,
        method=gen_methods.generate_rb_maxclique, n=9, a=0.8, p=0.25, r=2.7808)
