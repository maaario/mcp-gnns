import os

import gen_methods

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "generated", "small_dimacs_bhoslib")

if __name__ == "__main__":
    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "C.20.9"), num_graphs=100, seed=147,
        method=gen_methods.generate_gnp_graph, num_vertices=20, probability=0.9)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "C.20.5"), num_graphs=100, seed=148,
        method=gen_methods.generate_gnp_graph, num_vertices=20, probability=0.5)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "dsjc20"), num_graphs=100, seed=149,
        method=gen_methods.generate_dsjc_random_k, num_vertices=20)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "brock20"), num_graphs=100, seed=159,
        method=gen_methods.generate_brockington_random, num_vertices=20)

    gen_methods.generate_multiple_graphs(
        output_dir=os.path.join(DATA_DIR, "rb5-4"), num_graphs=100, seed=152,
        method=gen_methods.generate_rb_maxclique, n=5, a=0.8, p=0.25, r=2.7808)
