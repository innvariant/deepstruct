import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch import nn
from deepstruct.flexible_transform import GraphTransform
import torch.nn.functional as F

from deepstruct.traverse_strategies import FXTraversal


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


def test_transformer():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor)
    graph_transformer.transform(net)
    plot_graph(graph_transformer.get_graph(), "Transformation")


def test_fx_transformer():
    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(net)
    plot_graph(graph_transformer.get_graph(), "Transformation")
