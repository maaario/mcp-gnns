from collections import OrderedDict
import jsonpickle
import os

import networkx as nx
import numpy as np
import pandas as pd

from clique_finding_models.graph_dataset import load_graph_from_in, prepare_data_paths


def get_exp_ids_from_sacred_dir(sacred_dir):
    """Returns a list of experiment ids from selected sacred dir."""
    def is_string_int(x):
        try:
            int(x)
            return True
        except:
            return False

    return sorted([int(x) for x in os.listdir(sacred_dir) if is_string_int(x)])


def load_experiment_results(exp_dir):
    """Returns a config dict and a dataframe with metrics from experiment's directory."""
    with open(os.path.join(exp_dir, "config.json"), "r") as config_file:
        config = jsonpickle.decode(config_file.read())

    with open(os.path.join(exp_dir, "run.json"), "r") as run_file:
        run = jsonpickle.decode(run_file.read())
    config["status"] = run.get("status", "")

    with open(os.path.join(exp_dir, "metrics.json"), "r") as metrics_file:
        metrics = jsonpickle.decode(metrics_file.read())
        metrics_df = pd.DataFrame(columns=metrics.keys())
        for metric in metrics:
            try:
                metrics_df[metric] = metrics[metric]["values"]
            except Exception as e:
                if config.get("status", "") == "COMPLETED":
                    raise e

    return config, metrics_df


def process_exp_results_to_df(exp_ids, exp_results, hparams=list()):
    assert len(exp_results) == len(exp_ids), "Lengths of results and ids do not match."

    records = []

    for exp_id, exp_result in zip(exp_ids, exp_results):
        config, metrics = exp_result

        all_hparams = config.get("model_hparams", dict())
        all_hparams.update(config.get("trainer_hparams", dict()))
        filtered_hparams = {k: v for k, v in all_hparams.items() if k in hparams}

        record = OrderedDict(
            exp_id = exp_id,
            status = config.get("status", ""),
            data_set = os.path.basename(config.get("data_dir", "")),
            model = config.get("model", ""),
            train = config.get("train", None),
            epochs = metrics.shape[0],
            tag = config.get("tag", None),
            batch_size = config.get("batch_size", None),
            transform_y = config.get("transform_y", None),
        )
        record.update(filtered_hparams)
        try:
            record.update(metrics.tail(1).to_dict("records")[0])
        except Exception as e:
            if config.get("status", "") == "COMPLETED":
                raise e
        records.append(record)

    return pd.DataFrame.from_records(records)


def load_graph_to_nx(graph_path, clique_sizes_path=None):
    num_nodes, edge_list, clique_sizes = load_graph_from_in(graph_path, clique_sizes_path)

    graph = nx.Graph()
    graph.add_nodes_from(list(range(num_nodes)))
    graph.add_edges_from(edge_list)
    graph.clique_sizes = clique_sizes.numpy()

    return graph


def load_graph_dataset(data_dir, compute_neighborhood_sums=False):
    paths, train_mask, val_mask = prepare_data_paths(data_dir)

    columns = ["data_set", "graph_id", "graph_list_link", "train",
               "node_id", "degree", "clique_size"]
    if compute_neighborhood_sums:
        columns += ["sum_degrees", "sum_cliques"]
    values = list()

    graphs = []
    for i, path_pair in enumerate(paths):
        graph_path, clique_sizes = path_pair
        graph = load_graph_to_nx(graph_path, clique_sizes)
        graphs.append(graph)

        data_set = os.path.basename(os.path.dirname(graph_path))
        graph_id = int(os.path.splitext(os.path.basename(graph_path))[0])
        graph_list_link = i
        train = train_mask[i].item()

        node_info = np.array(list(zip(
            range(graph.number_of_nodes()),
            map(lambda x: x[1], graph.degree()),
            graph.clique_sizes
        )))
        for node_id, degree, clique_size in node_info:
            row = [data_set, graph_id, graph_list_link, train,
                   node_id, degree, clique_size]
            if compute_neighborhood_sums:
                neighbours = list(graph.neighbors(node_id))
                sum_of_degrees = node_info[neighbours][:, 1].sum()
                sum_of_cliques = node_info[neighbours][:, 2].sum()
                row += [sum_of_degrees, sum_of_cliques]
            values.append(row)

    graphs_df = pd.DataFrame(values, columns=columns)

    return graphs, graphs_df


def plot_graph(graph, graph_data, axes, color_col="abs_error", label_col="clique_size",
               neighobourhood_of_node=None, label="", **kwargs):
        graph_data = graph_data.reset_index()
        max_err = graph_data["abs_error"].max()
        data_set = graph_data.iloc[0]["data_set"]
        graph_id = graph_data.iloc[0]["graph_id"]
        if label == "":
            label = "{}-{}-{:.2f}".format(data_set, graph_id, max_err)

        if neighobourhood_of_node is not None:
            node = neighobourhood_of_node
            subgraph_nodes = [node] + list(graph.neighbors(node))
            graph_data = graph_data.iloc[subgraph_nodes]
            graph = graph.subgraph(subgraph_nodes)

        labels = {row[1][0]: "{:.1f}".format(row[1][1])
                  for row in graph_data[["node_id", label_col]].iterrows()}
        colors = [graph_data.loc[v][color_col] for v in graph.nodes]

        nx.draw_networkx(graph, with_labels=True,
                         node_color=colors, labels=labels, node_size=500,
                         ax=axes, **kwargs)
        axes.annotate(label, xy=(0, 1), xycoords="axes fraction")
        axes.axis("off")
