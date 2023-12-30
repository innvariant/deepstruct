import random

import networkx as nx

import torchvision.models
from matplotlib import pyplot as plt

from deepstruct.flexible_transform import GraphTransform
import torch.nn.functional as F

from deepstruct.node_map_strategies import LowLevelNodeMap
from deepstruct.traverse_strategies import FXTraversal, FrameworkTraversal

import torch
import torch.nn as nn
import torchvision.models as models


def plot_graph(graph, title):
    labels = nx.get_node_attributes(graph, 'name')
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(graph, labels=labels, with_labels=True, node_size=700, node_color='lightblue', font_size=8,
            ax=ax)
    plt.title(title)
    plt.show()


def calculate_network_metrics(G):
    metrics = {
        'Anzahl der Knoten': G.number_of_nodes(),
        'Anzahl der Kanten': G.number_of_edges(),
        'Durchschnittsgrad': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'Dichte': nx.density(G),
        'Durchschnittlicher Clusterkoeffizient': nx.average_clustering(G)
    }
    return metrics


class CNNtoRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(CNNtoRNN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        # CNN-Feature-Extraktion
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = self.avgpool(c_out)
        c_out = c_out.view(batch_size, timesteps, -1)
        r_out, (hn, cn) = self.rnn(c_out)
        r_out2 = self.fc(r_out[:, -1, :])
        return r_out2


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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=6, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
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


#def test_framework_transformer():
#    net = SimpleCNN()
#    input_tensor = torch.randn(1, 1, 6, 6)
#    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FrameworkTraversal())
#    graph_transformer.transform(net)
#    plot_graph(graph_transformer.get_graph(), "Transformation")


def test_fx_transformer_simple_network():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
    simplenet_names = ["cos", "fc", "size", "view", "pool", "relu", "conv1", "x", "output"]
    assert len(graph.nodes) == len(simplenet_names)
    graph_names = nx.get_node_attributes(graph, 'name').values()
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
    graph_names = nx.get_node_attributes(graph, 'name').values()
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
    graph_names = nx.get_node_attributes(graph, 'name').values()
    assert all(name in graph_names for name in test_names)
    assert all(name in test_names for name in graph_names)


def test_fx_transformer_simple_network_includes_excludes():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        exclude_fn=[torch.cos, torch.Tensor.size],
        include_modules=[torch.nn.modules.Linear, torch.nn.modules.Conv2d]))
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation")
    test_names = ["fc", "view", "relu", "conv1", "x", "output"]
    assert len(graph.nodes) == len(test_names)
    graph_names = nx.get_node_attributes(graph, 'name').values()
    assert all(name in graph_names for name in test_names)
    assert all(name in test_names for name in graph_names)


def test_fx_fold_modules():
    resnet = torchvision.models.resnet18(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        fold_modules=[torch.nn.modules.Sequential]))
    graph_transformer.transform(resnet)
    graph = graph_transformer.get_graph()
    assert len(graph.nodes) == 13
    plot_graph(graph, "Transformation resnet")


def test_fx_unfold_modules():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        unfold_modules=[torch.nn.modules.Linear]
    ))
    graph_transformer.transform(net)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation simplenet")


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


def test_fx_resnet18():
    print(" ")
    resnet = torchvision.models.resnet18(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor,
                                       traversal_strategy=FXTraversal(),
                                       node_map_strategy=LowLevelNodeMap())
    graph_transformer.transform(resnet)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation resnet")


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


def test_fx_alexnet():
    print(" ")
    alexnet = torchvision.models.alexnet(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor,
                                       traversal_strategy=FXTraversal(
                                           unfold_modules=[object],
                                           fold_modules=[torch.nn.Conv2d,
                                                         torch.nn.Linear,
                                                         torch.nn.BatchNorm2d])
                                       )
    graph_transformer.transform(alexnet)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation alexnet")


def test_fx_resnet50():
    print(" ")
    res34 = torchvision.models.resnet50(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(unfold_modules=[object],
                                                                                    fold_modules=[torch.nn.Conv2d,
                                                                                                  torch.nn.Linear,
                                                                                                  torch.nn.BatchNorm2d])
                                       )
    graph_transformer.transform(res34)
    plot_graph(graph_transformer.get_graph(), "Transformation resnet 50")


def test_fx_googlenet():
    print(" ")
    gnet = torchvision.models.googlenet(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(
        unfold_modules=[object],
        fold_modules=[torch.nn.Conv2d,
                      torch.nn.Linear,
                      torch.nn.BatchNorm2d,
                      torch.nn.MaxPool2d,
                      torch.nn.AdaptiveAvgPool2d]), node_map_strategy=LowLevelNodeMap()
                                       )
    graph_transformer.transform(gnet)
    plot_graph(graph_transformer.get_graph(), "Transformation googlenet")


def test_fx_hybridmodel():
    print(" ")
    hybridmodel = CNNtoRNN(num_classes=10, hidden_size=256, num_layers=2)
    input_tensor = torch.rand(4, 10, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal(),
                                       node_map_strategy=LowLevelNodeMap())
    graph_transformer.transform(hybridmodel)
    plot_graph(graph_transformer.get_graph(), "Transformation hybridmodel")


def test_ged():
    net1 = SimpleCNN()
    net2 = SmallCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer2 = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(net1)
    graph_transformer2.transform(net2)
    G1 = graph_transformer.get_graph()
    G2 = graph_transformer2.get_graph()

    metrics_G1 = calculate_network_metrics(G1)
    metrics_G2 = calculate_network_metrics(G2)
    ged = nx.graph_edit_distance(G1, G2)
    print("Kennzahlen von G1:", metrics_G1)
    print("Kennzahlen von G2:", metrics_G2)
    print(ged)

