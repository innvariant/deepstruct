import random

import networkx as nx
import torch
import torchvision.models
from matplotlib import pyplot as plt
from torch import nn
from deepstruct.flexible_transform import GraphTransform
import torch.nn.functional as F

from deepstruct.traverse_strategies import FXTraversal, FrameworkTraversal


def plot_graph(graph, title):
    labels = {node: graph.nodes[node]['name'] for node in graph.nodes()}
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(graph, labels=labels, with_labels=True, node_size=700, node_color='lightblue', font_size=8,
            ax=ax)
    plt.title(title)
    plt.show()


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


class FuncConstructorCNN(nn.Module):
    def __init__(self):
        super(FuncConstructorCNN, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        ones_tensor = torch.ones_like(x)
        y = torch.ones(10) + ones_tensor
        return self.fc(y)


class ControlFlowCNN(nn.Module):
    def __init__(self):
        super(ControlFlowCNN, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.rand_num = random.randint(0, 10)

    def forward(self, x):
        if x.sum() > 0:
            x = self.fc(x)
        else:
            x = F.relu(self.fc(x))

        if self.rand_num < 5:
            x = F.sigmoid(x)
        return x


def test_transformer():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FrameworkTraversal())
    graph_transformer.transform(net)
    plot_graph(graph_transformer.get_graph(), "Transformation")


def test_fx_transformer_simple_network():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
    simplenet_names = ["cos", "fc", "size", "view", "pool", "relu", "conv1", "x", "output"]
    assert len(graph.nodes) == len(simplenet_names)
    graph_names = dict(graph.nodes(data='name')).values()
    assert all(name in graph_names for name in simplenet_names)
    assert all(name in simplenet_names for name in graph_names)


def test_fx_functional_constructor():
    net = FuncConstructorCNN()
    input_tensor = torch.randn(1, 10)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(net)
    plot_graph(graph_transformer.get_graph(), "Transformation")


def test_fx_control_flow():
    net = ControlFlowCNN()
    input_tensor = torch.randn(1, 10)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(net)
    plot_graph(graph_transformer.get_graph(), "Transformation")


def test_fx_transformer_simple_network_excludes():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    excluded_functions = [torch.nn.functional.relu, torch.cos]
    excluded_modules = [torch.nn.modules.Linear, torch.nn.modules.MaxPool2d]
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        exclude_fn=excluded_functions,
        exclude_modules=excluded_modules))
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
    simplenet_names = ["cos", "fc", "size", "view", "pool", "relu", "conv1", "x", "output"]
    test_names = ["size", "view", "conv1", "x", "output"]
    assert len(graph.nodes) == len(simplenet_names) - len(excluded_modules) - len(excluded_functions)
    graph_names = dict(graph.nodes(data='name')).values()
    assert all(name in graph_names for name in test_names)
    assert all(name in test_names for name in graph_names)


def test_fx_transformer_simple_network_includes():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    included_functions = [torch.cos]
    included_modules = [torch.nn.modules.Conv2d, torch.nn.modules.MaxPool2d]
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        include_fn=included_functions,
        include_modules=included_modules))
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
    test_names = ["cos", "pool", "conv1", "x", "output"]
    assert len(graph.nodes) == len(included_functions) + len(included_modules) + 2  # count input and output
    graph_names = dict(graph.nodes(data='name')).values()
    assert all(name in graph_names for name in test_names)
    assert all(name in test_names for name in graph_names)


def test_fx_transformer_simple_network_includes_excludes():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        exclude_fn=[torch.cos, torch.Tensor.size],  # TODO: size still included
        include_modules=[torch.nn.modules.Linear, torch.nn.modules.Conv2d]))
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
    test_names = ["fc", "view", "relu", "conv1", "x", "output"]
    assert len(graph.nodes) == len(test_names)
    graph_names = dict(graph.nodes(data='name')).values()
    assert all(name in graph_names for name in test_names)
    assert all(name in test_names for name in graph_names)


def test_fx_resnet18():
    resnet = torchvision.models.resnet18(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        include_fn=[torch.Tensor.add, torch.add]))
    graph_transformer.transform(resnet)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
