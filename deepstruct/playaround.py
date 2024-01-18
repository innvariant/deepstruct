import torch
import networkx as nx
import torchvision.models
from torch import nn
import matplotlib.pyplot as plt


class GraphTransformer:
    def __init__(self, shown_modules=None, split_modules=None, ignored_modules=None):
        self.shown_modules = shown_modules
        self.split_modules = split_modules
        self.ignored_modules = ignored_modules if ignored_modules is not None else []
        self.pos = {}
        self.graph = nx.DiGraph()
        self.next_node_index = 0

    def transform(self, model: torch.nn.Module, depth=1):
        for module in model.children():
            self.classify_module(module, depth)

    def classify_module(self, module, depth):
        name = type(module).__name__
        if any(isinstance(module, i) for i in self.ignored_modules):
            return
        if self.split_modules is None or any(isinstance(module, i) for i in self.split_modules):
            for child in module.children():
                self.classify_module(child, depth)
        if self.shown_modules is None or any(isinstance(module, i) for i in self.shown_modules):
            self._add_node(name)
        else:
            print("This module is not considered: " + name)

    def _add_node(self, name):
        current_node = self.next_node_index
        prev_node = self.next_node_index - 1
        self.graph.add_node(current_node, name=name)
        nodes = self.graph.nodes()
        if prev_node in nodes and current_node in nodes:
            self.graph.add_edge(prev_node, current_node)
        self.pos[current_node] = (0, current_node)
        self.next_node_index += 1



class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU()
            ),
            nn.Linear(512, 10),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(3, 3, 20)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


custom_model = CustomNN()
print(custom_model.__dict__.get("_modules"))


def plot_graph(graph, title, positions):
    labels = {node: graph.nodes[node]['name'] for node in graph.nodes()}
    fig, ax = plt.subplots(figsize=(5, 12))
    nx.draw(graph, positions, labels=labels, with_labels=True, node_size=900, node_color='lightblue', font_size=8,
            ax=ax)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    resnet = torchvision.models.resnet18(True)
    shown_modules = [nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.BatchNorm2d, nn.Linear, nn.Flatten]
    split_modules = [nn.Sequential, nn.ModuleList, nn.ModuleDict]
    graph_transformer = GraphTransformer()
    graph_transformer.transform(custom_model)
    print(graph_transformer.graph)
    print(graph_transformer.pos)
    plot_graph(graph_transformer.graph, "High Level Extraction", graph_transformer.pos)
