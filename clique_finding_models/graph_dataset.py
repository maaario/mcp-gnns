import os

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


def load_graph_from_in(graph_path, clique_sizes_path=None):
    """
    Loads graph from .in file and returns it as Data object, optionally also reads clique sizes.
    """
    with open(graph_path, "r") as f:
        num_nodes, num_edges = [int(x) for x in f.readline().strip().split()]
        edge_list = []
        for i in range(num_edges):
            node1, node2 = [int(x) - 1 for x in f.readline().strip().split()]
            edge_list.append((node1, node2))
            edge_list.append((node2, node1))

    if clique_sizes_path is not None:
        with open(clique_sizes_path, "r") as f:
            clique_sizes = torch.tensor(
                [int(x) for x in f.readline().strip().split()], dtype=torch.float)
    else:
        clique_sizes = torch.ones(num_nodes, dtype=torch.float)

    return num_nodes, edge_list, clique_sizes


def load_graph_to_data(graph_path, clique_sizes_path=None):
    num_nodes, edge_list, clique_sizes = load_graph_from_in(graph_path, clique_sizes_path)

    return Data(
        x=torch.ones([num_nodes, 1]),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        y=clique_sizes,
    )


def prepare_data_paths(path):
    """
    Recursively searches path for .in files in sub-directories.
    Returns a list of pairs (graph file path, clique file path).
    Also returns a training and validation mask splitting graphs in 80:20 ratio, keeping the
    ratio constant for each subdirectory.
    """
    assert os.path.exists(path) and os.path.isdir(path), \
        "Data directory '{}'does not exist!".format(path)

    files = os.listdir(path)
    graphs = sorted([os.path.join(path, f) for f in files if f.endswith(".in")])
    clique_sizes = sorted([os.path.join(path, f) for f in files if f.endswith(".out")])
    assert len(graphs) == len(clique_sizes), "Counts of .in and .out files do not correspond!"
    path_pairs = list(zip(graphs, clique_sizes))

    train_mask = torch.zeros(len(path_pairs), dtype=torch.uint8)
    val_mask = torch.zeros(len(path_pairs), dtype=torch.uint8)
    train_val_split = int(0.8 * len(graphs))
    train_mask[:train_val_split] = 1
    val_mask[train_val_split:] = 1
    train_masks, val_masks = [train_mask], [val_mask]

    subdirectories = sorted([d for d in files if os.path.isdir(os.path.join(path, d))])
    for subdir in subdirectories:
        pp, tm, vm = prepare_data_paths(os.path.join(path, subdir))
        path_pairs.extend(pp)
        train_masks.append(tm)
        val_masks.append(vm)

    return path_pairs, torch.cat(train_masks), torch.cat(val_masks)


class GraphDataset(InMemoryDataset):
    """
    Processes graphs from 'source_data_dir' into 'root' directory.
    This class is intended for training datasets, so the expected outputs
    (clique sizes in .out file) must be present for each graph.
    Splits graphs from each subdirectory into train and validation data with 80:20 split.
    Does not download / copy graphs from their source dir, only process them to .pt files.
    """
    def __init__(self, root, source_data_dir, train=True, full=False, transform=None,
                 pre_transform=None):
        self.source_data_dir = source_data_dir
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        if full:
            self.processed_data_path = self.processed_paths[2]
        else:
            self.processed_data_path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(self.processed_data_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["train.pt", "validate.pt", "full.pt", "train_paths.txt", "val_paths.txt"]

    def download(self):
        pass

    def process(self):
        paths, train_mask, val_mask = prepare_data_paths(self.source_data_dir)
        graph_list = []

        for graph_path, clique_sizes_path in paths:
            graph_list.append(load_graph_to_data(graph_path, clique_sizes_path))

        with open(self.processed_paths[3], "w") as log_train:
            with open(self.processed_paths[4], "w") as log_val:
                for i, path_pair in enumerate(paths):
                    graph_path, clique_sizes_path = path_pair
                    if train_mask[i]:
                        log_train.write(graph_path + "\n" + clique_sizes_path + "\n")
                    if val_mask[i]:
                        log_val.write(graph_path + "\n" + clique_sizes_path + "\n")

        if self.pre_filter is not None:
            graph_list = [data for data in graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(data) for data in graph_list]

        train_list = [g for i, g in enumerate(graph_list) if train_mask[i]]
        val_list = [g for i, g in enumerate(graph_list) if val_mask[i]]

        # Save training data.
        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])

        # Save validation data.
        data, slices = self.collate(val_list)
        torch.save((data, slices), self.processed_paths[1])

        # Save full data.
        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[2])
