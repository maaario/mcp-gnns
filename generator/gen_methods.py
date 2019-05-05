import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import networkx as nx
import numpy as np
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CLIQUE_FOR_EACH_VERTEX_PATH = os.path.join(PROJECT_DIR, "maxclique", "main_clique_for_each_vertex")
CONVERT_DIMACS_TO_IN_PATH = os.path.join(PROJECT_DIR, "generator", "convert_dimacs_to_in.sh")
CONVERT_BIN_TO_DIMACS_PATH = os.path.join(
    PROJECT_DIR, "generator", "third_party", "ANSI", "bin2asc")
BROCKINGTON_GENERATOR_PATH = os.path.join(
    PROJECT_DIR, "generator", "third_party", "brockington", "graphgen")
RB_GENERATOR_PATH = os.path.join(
    os.path.join(PROJECT_DIR, "generator", "third_party", "RB", "generator", "rbGenerator.py"))


def compute_labels(input_path, output_path):
    """Computes the sizes of largest clique for each vertex for graph in .in format."""
    with open(input_path, "r") as input_file, open(output_path, "w") as output_file:
        subprocess.check_call(CLIQUE_FOR_EACH_VERTEX_PATH, stdin=input_file, stdout=output_file)


def convert_dimacs_to_in(input_path, output_path):
    """Converts graph from dimacs .clq format to .in format."""
    subprocess.check_call([CONVERT_DIMACS_TO_IN_PATH, input_path, output_path])


def convert_bin_to_dimacs(input_path, output_path):
    """Converts graph from dimacs .clq format to .in format."""
    subprocess.check_call([CONVERT_BIN_TO_DIMACS_PATH, input_path, output_path])


def load_networkx_graph_from_in(path):
    """Loads a networkx graph from a file in .in format."""
    def process_line_to_int_pair(line):
        line = line.strip().split()
        return int(line[0]) - 1, int(line[1]) - 1

    with open(path, "r") as f:
        lines = f.readlines()
        n, m = process_line_to_int_pair(lines[0])
        edges = [process_line_to_int_pair(line) for line in lines][1:]

    graph = nx.Graph()
    graph.add_nodes_from(range(1, n + 1))
    graph.add_edges_from(edges)
    return graph


def save_networkx_graph_to_in(graph, path):
    """Writes a networkx graph into a file using .in format."""
    with open(path, "w") as f:
        f.write("{} {}\n".format(len(graph.nodes), len(graph.edges)))
        f.write("\n".join(["{} {}".format(e1 + 1, e2 + 1) for e1, e2 in graph.edges]))


def generate_multiple_graphs(output_dir, num_graphs, method, seed=0, labels=True, **kwargs):
    """
    If the output directory exists, data generation is skipped.
    Otherwise `method` is called with output file name, seed and kwargs as arguments.
    After the graphs are generated, labels (clique sizes) are computed.
    """
    if os.path.exists(output_dir):
        print("Directory {} already exits, inputs won't be generated.".format(output_dir),
              file=sys.stderr)
        return
    os.makedirs(output_dir)

    needs_temp_dir = getattr(method, "needs_temp_dir", False)
    if needs_temp_dir:
        temp_dir = tempfile.mkdtemp()
        print("Creating temporary directory {} (will be deleted after generation)".format(temp_dir),
              file=sys.stderr)
        kwargs["temp_dir"] = temp_dir

    rng = np.random.RandomState(seed=seed)

    final_folder = os.path.basename(os.path.normpath(output_dir))
    for i in tqdm(range(num_graphs), desc=final_folder):
        file_path_base = os.path.join(output_dir, "{:05d}".format(i))
        graph_file_path = file_path_base + ".in"
        seed_i = rng.randint(0, 2**32)

        random.seed(seed_i)
        np.random.seed(seed_i)
        method(path=graph_file_path, seed=seed_i, **kwargs)

        if labels:
            labels_file_path = file_path_base + ".out"
            compute_labels(graph_file_path, labels_file_path)

    if needs_temp_dir:
        shutil.rmtree(temp_dir)


def store_parameters(path, **kwargs):
    """Stores all keyword arguments into json file at path.json"""
    with open(path + ".json", "w") as f:
        json.dump(kwargs, f, indent=2)


def generate_gnp_graph(path, seed, num_vertices, probability):
    """Generate random graph with each edge of probability `probability`."""
    graph = nx.gnp_random_graph(num_vertices, probability, seed=seed)
    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_gnp_graph.__name__,
                     seed=seed, num_vertices=num_vertices, probability=probability)


def generate_gnp_graph_random_p(path, seed, num_vertices):
    probability = random.random()
    generate_gnp_graph(path, seed, num_vertices, probability)


def generate_gnp_graph_random_p_denser(path, seed, num_vertices):
    probability = random.random() ** 0.5
    generate_gnp_graph(path, seed, num_vertices, probability)


def generate_regular_graph(path, seed, num_vertices, degree):
    """ 
    Generate random regular graph.
    Counts of regular graphs for various n, d: http://mathworld.wolfram.com/RegularGraph.html
    """
    graph = nx.random_regular_graph(degree, num_vertices, seed=seed)
    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_regular_graph.__name__,
                     seed=seed, num_vertices=num_vertices, degree=degree)


def generate_graph_with_various_clique_sizes(path, seed, num_vertices):
    """
    Sparse gnp graph (p from (0, 0.5)) with added cliques of sizes 4 .. n/2 so that the sum of their
    sizes is at most n.
    """
    def generate_clique_sizes(min_clique_size=4, max_clique_size=num_vertices / 2):
        clique_sizes = []
        remaining_vertices = num_vertices
        while remaining_vertices >= min_clique_size:
            clique_size = random.randint(
                min_clique_size, min((max_clique_size, remaining_vertices)))
            remaining_vertices -= clique_size
            clique_sizes.append(clique_size)
        return clique_sizes

    probability = random.random() * 0.5
    graph = nx.gnp_random_graph(num_vertices, probability, seed=seed)

    clique_sizes = generate_clique_sizes()
    for clique_size in clique_sizes:
        clique = nx.complete_graph(clique_size)
        vertices = random.sample(range(0, num_vertices), clique_size)
        nx.relabel_nodes(clique, dict(zip(range(clique_size), vertices)), copy=False)
        graph.add_edges_from(clique.edges)

    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_graph_with_various_clique_sizes.__name__,
                     seed=seed, num_vertices=num_vertices,
                     other=dict(min_clique_size=4, max_clique_size=num_vertices / 2))


def generate_dsjc(path, seed, num_vertices, k):
    """
    Cooked graphs from Optimization by simulated annealing: an experimental evaluation; part {II},
    DS Johnson, 1991:
        1) split vertices to k color classes
        2) edges to pairs from different colors with p = k/(2(k-1)) => avg degree n/2
           (max edges n * (k-1)/k for 1 vertex, wanted = p * max => p = wanted / max)
        3) select representant of each color & add one K-clique
    """
    coloring = np.random.randint(0, k, num_vertices)
    _, partition = np.unique(coloring, return_counts=True)
    used_k = len(partition)
    graph = nx.random_partition_graph(
        partition, p_in=0, p_out=1. * used_k / (2 * (used_k - 1)), seed=seed)
    representants = np.cumsum(partition) - 1
    clique = nx.complete_graph(used_k)
    nx.relabel_nodes(clique, dict(zip(range(used_k), representants)), copy=False)

    graph.add_edges_from(clique.edges)
    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_dsjc.__name__,
                     seed=seed, num_vertices=num_vertices, k=k, 
                     other=dict(partition=partition.tolist()))


def generate_dsjc_random_k(path, seed, num_vertices, max_k=None):
    if max_k is None:
        max_k = num_vertices
    k = random.randint(2, max_k)
    generate_dsjc(path, seed, num_vertices, k)


def generate_brockington(path, seed, temp_dir, num_vertices, clique_size, density, depth):
    # Set up parameters.
    g = "-g{}".format(num_vertices)
    c = "-c{}".format(clique_size)
    p = "-p{:.6f}".format(density)
    d = "-d{}".format(depth)
    s = "-s{}".format(seed)

    # Write the name of the output file to a temporary file.
    output_definition_file_path = os.path.join(temp_dir, "output_definition_file")
    with open(output_definition_file_path, "w") as output_definition_file:
        temp_graph_path = os.path.join(temp_dir, "graph")
        output_definition_file.write(temp_graph_path)

    # Generate graph.
    with open(output_definition_file_path, "r") as output_definition_file:
        with open(os.devnull, "w") as devnull:
            subprocess.check_call(
                [BROCKINGTON_GENERATOR_PATH, g, c, p, d, s],
                stdin=output_definition_file, stdout=devnull)

    # Convert to .in format.
    bin_file_path = os.path.join(temp_dir, "graph.b")
    dimacs_file_path = os.path.join(temp_dir, "graph.clq")
    convert_bin_to_dimacs(bin_file_path, dimacs_file_path)
    convert_dimacs_to_in(dimacs_file_path, path)
    store_parameters(path, method=generate_brockington.__name__,
                     seed=seed, num_vertices=num_vertices, clique_size=clique_size,
                     density=density, depth=depth)
generate_brockington.needs_temp_dir = True


def generate_brockington_random(path, seed, temp_dir, num_vertices, max_tries=10, max_clq=None):
    if max_clq is None:
        max_clq = num_vertices

    for try_counter in range(max_tries):
        try:
            clique_size = random.randint(3, max_clq)
            density = random.random()
            depth = random.randint(0, 4)
            generate_brockington(path, seed, temp_dir, num_vertices, clique_size, density, depth)
            break
        except subprocess.CalledProcessError as e:
            if try_counter < max_tries - 1:
                continue
            else:
                raise e
generate_brockington_random.needs_temp_dir = True


def generate_hamming(path, seed, dimension, distance):
    """
    Generate graph of `dimension`-bit words with an edge between words if and only if
    the two words are at least hamming distance `distance` apart.
    """
    labels = np.arange(2**dimension, dtype=np.uint32).reshape((2**dimension, 1)).view(np.uint8)
    words = np.unpackbits(labels, axis=1)
    distances = words.dot(1 - words.T) + (1 - words).dot(words.T)
    graph = nx.from_numpy_matrix(distances >= distance)
    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_hamming.__name__,
                     seed=seed, dimension=dimension, distance=distance)


def precomute_hamming_graph_params():
    params = []
    for dimension in range(20):
        for distance in range(1, dimension + 1):
            params.append((dimension, distance))
    return params


def generate_hamming_seeded(path, seed, max_dimension):
    """With increasing seed generates hamming graphs (1, 1), (2, 1), (2, 2), (3, 1)..."""
    num_various_graphs = (max_dimension + 1) * max_dimension / 2
    graph_id = int(seed % num_various_graphs)
    dimension, distance = generate_hamming_seeded.precomputed_params[graph_id]
    generate_hamming(path, seed, dimension, distance)
generate_hamming_seeded.precomputed_params = precomute_hamming_graph_params()


def generate_rb_maxclique(path, seed, temp_dir, n, a, p, r):
    """
    num_vertices = n * n^a (n cliques with n^a vertices = n variables with domain sizes n^a)
    num_edges = p * n^{2a} * r * n * ln(n) (r n ln n constraints for a variable pair)
    Run generator for MIS/MVC and compute the complement graph.
    """
    rb_generator_output_path = os.path.join(temp_dir, "vc.clq")
    with open(rb_generator_output_path, "w") as vc_file:
        subprocess.check_call(
            ["python2", RB_GENERATOR_PATH, "-e", "VC", "-s", str(seed),
             str(n), str(a), str(p), str(r)], stdout=vc_file)

    vc_in_file_path = os.path.join(temp_dir, "vc.in")
    convert_dimacs_to_in(rb_generator_output_path, vc_in_file_path)

    graph = load_networkx_graph_from_in(vc_in_file_path)
    graph = nx.complement(graph)
    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_rb_maxclique.__name__, seed=seed, n=n, a=a, p=p, r=r)
generate_rb_maxclique.needs_temp_dir = True


def generate_simple_one_clique(path, seed, num_vertices, probability):
    """A random graph with edge density `probability` with one clique added."""
    graph = nx.gnp_random_graph(num_vertices, probability, seed=seed)
    clique_size = random.randint(1, num_vertices)
    clique = nx.complete_graph(clique_size)
    graph.add_edges_from(clique.edges)

    save_networkx_graph_to_in(graph, path)
    store_parameters(path, method=generate_gnp_graph.__name__,
                     seed=seed, num_vertices=num_vertices, probability=probability)
