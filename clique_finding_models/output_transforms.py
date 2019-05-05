"""
Defines output transforms and their reverse operations, enabling computation of absolute metrics.

transform_y_<type>(data) -> transformed_data
    These functions can be passed to dataset object to transform the ground truth clique sizes 
    to specific output format.

transform_y_<type>.reverse(transformed_data, transformed_y) -> original_y
    These functions can be used to compute the (absolute) clique sizes from (relative) predicted
    clique sizes, and also to gain absolute clique sizes from transformed ground truth data.
    Useful for computation of metrics based on absolute clique sizes.
"""
import torch
from torch_geometric.utils import degree


def transform_y_none(data):
    return data


def transform_y_none_reverse(data, transformed_y):
    return transformed_y

transform_y_none.reverse = transform_y_none_reverse


def transform_y_relative_to_max_clique(data):
    data.max_clique = torch.ones((data.num_nodes,), device=data.y.device) * data.y.max()
    data.y = data.y / data.max_clique
    return data


def transform_y_relative_to_max_clique_reverse(data, transformed_y):
    return transformed_y * data.max_clique

transform_y_relative_to_max_clique.reverse = transform_y_relative_to_max_clique_reverse


def transform_y_relative_to_degree(data):
    data.degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)
    data.y = data.y / (data.degrees + 1)
    return data


def transform_y_relative_to_degree_reverse(data, transformed_y):
    return transformed_y * (data.degrees + 1)

transform_y_relative_to_degree.reverse = transform_y_relative_to_degree_reverse


def transform_y_binary(data):
    data.y = (data.y == data.y.max()).float()
    return data

"""
The purpose of this dict is to enable to set the output transfrom using only a string. 
Useful for sacred configuration and commandline arguments.
"""
transform_y_dict = dict(
    none = transform_y_none,
    relative_to_max_clique = transform_y_relative_to_max_clique,
    relative_to_degree = transform_y_relative_to_degree,
    binary=transform_y_binary,
)
