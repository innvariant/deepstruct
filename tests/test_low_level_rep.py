import networkx as nx
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from deepstruct.flexible_transform import GraphTransform
import torch.nn.functional as F

from deepstruct.node_map_strategies import LowLevelNodeMap
from deepstruct.transform import Conv2dLayerFunctor
from deepstruct.traverse_strategies import FXTraversal

import torch
import torch.nn as nn


def plot_graph(graph, title):
    labels = nx.get_node_attributes(graph, 'name')
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(graph, labels=labels, with_labels=True, node_size=700, node_color='lightblue', font_size=8,
            ax=ax)
    plt.title(title)
    plt.show()


def print_adjacent_matrix(graph):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    node_labels = [graph.nodes[n]['name'] for n in graph.nodes()]
    adj_matrix = nx.adjacency_matrix(graph)
    adj_df = pd.DataFrame(adj_matrix.todense(), index=node_labels, columns=node_labels)
    print(adj_df)


def calculate_network_metrics(graph):
    metrics = {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'avg degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        'density': nx.density(graph),
        'Average cluster coefficient': nx.average_clustering(graph)
    }
    return metrics


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(54, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.cos(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(9, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        x = torch.cos(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


def test_fx_low_level_linear_fully_connected():
    print(" ")
    net = SmallCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor,
                                       traversal_strategy=FXTraversal(),
                                       node_map_strategy=LowLevelNodeMap()
                                       )
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    fc_nodes = graph.get_indices_for_name('fc')
    fc2_nodes = graph.get_indices_for_name('fc2')
    cos_nodes = graph.get_indices_for_name('cos')
    assert len(fc_nodes) == 9
    assert len(fc2_nodes) == 6
    assert len(cos_nodes) == 3
    for node in fc_nodes:
        assert graph.in_degree(node) == 1
        assert graph.out_degree(node) == 6
    for node in fc2_nodes:
        assert graph.in_degree(node) == 9
        assert graph.out_degree(node) == 3
    for node in cos_nodes:
        assert graph.in_degree(node) == 6
        assert graph.out_degree(node) == 1

    plot_graph(graph, "Transformation smallnet")
    print_adjacent_matrix(graph)


def test_fx_low_level_linear():
    print(" ")
    net = SmallCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor,
                                       traversal_strategy=FXTraversal(),
                                       node_map_strategy=LowLevelNodeMap(0.15)
                                       )
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation smallnet")


def test_convnet():
    print(" ")
    conv_net = ConvNet()
    input_tensor = torch.rand(1, 1, 2, 2)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(),
                                       node_map_strategy=LowLevelNodeMap())
    graph_transformer.transform(conv_net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "convnet")
    conv_nodes = graph.get_indices_for_name('conv1')
    pool_nodes = graph.get_indices_for_name('pool')
    assert len(conv_nodes) == 4
    assert len(pool_nodes) == 18
    for node in conv_nodes:
        assert graph.in_degree(node) == 1
    in_deg_sum = 0
    nodes_with_indeg = 0
    for node in pool_nodes:
        in_deg_sum += graph.in_degree(node)
        if graph.in_degree(node) != 0:
            nodes_with_indeg += 1
        assert graph.out_degree(node) == 1
    assert in_deg_sum == len(pool_nodes)


def test_conv_simple():
    class ConvModel(nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            channels_in = 3
            kernel_size = (3, 3)
            self.conv1 = nn.Conv2d(
                in_channels=channels_in, out_channels=2, kernel_size=kernel_size, stride=1
            )

        def forward(self, x):
            return self.conv1(x)

    input_width = 5
    input_height = 5
    channels_in = 3
    model = ConvModel()
    model.conv1.weight[:, :].data += 10
    random_input = torch.rand(size=(1, channels_in, input_height, input_width))
    graph_transformer = GraphTransform(random_input, node_map_strategy=LowLevelNodeMap(0.01))

    graph_transformer.transform(model)
    graph = graph_transformer.get_graph()
    output = model.forward(random_input)
    number_output_features = np.prod(output.shape)
    plot_graph(graph, "conv simple")
    assert len(graph.get_indices_for_name('output')) == number_output_features


def test_conv_with_add():
    class ConvModel(nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        def forward(self, x):
            x = self.conv1(x)
            tmp = x
            x = self.pool(x)
            x = torch.cos(x)
            x = torch.add(x, tmp)
            return x

    print(" ")
    conv_model = ConvModel()
    input_tensor = torch.rand(1, 1, 2, 2)
    print(input_tensor)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(),
                                       node_map_strategy=LowLevelNodeMap())
    graph_transformer.transform(conv_model)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "convnet")
    conv_nodes = graph.get_indices_for_name('conv1')
    pool_nodes = graph.get_indices_for_name('pool')
    cos_nodes = graph.get_indices_for_name('cos')
    add_nodes = graph.get_indices_for_name('add')
    print(len(conv_nodes), len(pool_nodes), len(cos_nodes), len(add_nodes))
    for node in conv_nodes:
        assert graph.in_degree(node) == 1
    in_deg_sum = 0
    nodes_with_indeg = 0
    pools_with_zero_deg = 0
    for node in pool_nodes:
        in_deg_sum += graph.in_degree(node)
        if graph.in_degree(node) == 0:
            pools_with_zero_deg += 1
        if graph.in_degree(node) != 0:
            nodes_with_indeg += 1
        assert graph.out_degree(node) == 1
    print(pools_with_zero_deg)


def test_realistic_convolution():
    class ConvModel(nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            channels_in = 3
            kernel_size = (5, 5)
            self.conv1 = nn.Conv2d(
                in_channels=channels_in, out_channels=2, kernel_size=kernel_size, stride=1
            )

        def forward(self, x):
            return self.conv1(x)

    input_width = 100
    input_height = 100
    channels_in = 3
    model = ConvModel()
    model.conv1.weight[:, :].data += 10
    random_input = torch.rand(size=(1, channels_in, input_height, input_width))
    graph_transformer = GraphTransform(random_input, node_map_strategy=LowLevelNodeMap(0.01))

    graph_transformer.transform(model)
    graph = graph_transformer.get_graph()
    output = model.forward(random_input)
    number_output_features = np.prod(output.shape)
    # plot_graph(graph, "realistic conv")
    assert len(graph.get_indices_for_name('output')) == number_output_features


def test_realistic_convolution2():
    # Arrange
    input_width = 100  # 100x100 is already quite a huge graph
    input_height = 100
    channels_in = 3
    kernel_size = (5, 5)
    model = torch.nn.Conv2d(
        in_channels=channels_in, out_channels=2, kernel_size=kernel_size, stride=1
    )
    # Make sure each weight is large enough so none is getting "pruned"
    model.weight[:, :].data += 10
    output = model.forward(torch.rand(size=(1, channels_in, input_height, input_width)))
    number_output_features = np.prod(output.shape)

    functor = Conv2dLayerFunctor(input_width, input_height, threshold=0.01)

    # Act
    result = functor.transform(model)

    # Assert
    assert result.last_layer_size == number_output_features


def test_transposed_convolution():
    print(" ")
    # Arrange
    input_width = 2
    input_height = 2
    channels_in = 1
    model = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=1)
    # Make sure each weight is large enough so none is getting "pruned"
    model.weight[:, :].data += 10
    output = model.forward(torch.rand(size=(1, channels_in, input_height, input_width)))
    number_output_features = np.prod(output.shape)

    functor = Conv2dLayerFunctor(input_width, input_height, threshold=0.01)

    # Act
    result = functor.transform(model)
    print(number_output_features)
    # Assert
    assert result.last_layer_size == number_output_features
    nx.draw(result, with_labels=True, node_size=700, node_color='lightblue', font_size=8)
    plt.show()
