import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

import deepstruct.graph
import deepstruct.sparse
import deepstruct.util


class Foo(nn.Module):
    def __init__(self, num_classes: int, input_channels: int, input_size: int):
        super(Foo, self).__init__()

        self._num_classes = num_classes

        reduced_channels = 100
        reduction_steps = max(math.floor(math.log2(input_size)) - 5, 1)
        input_convs = [ReductionCell(input_channels, reduced_channels)]
        for step in range(1, reduction_steps):
            input_convs.append(ReductionCell(reduced_channels, reduced_channels))
        self._input_convs = nn.ModuleList(input_convs)

        self.conv1 = nn.Conv2d(reduced_channels, 100, (1, 1))
        self.bn1 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(100, self._num_classes)

    def forward(self, input):
        y = input
        for conv in self._input_convs:
            y = conv(y)

        y = self.conv1(y)
        y = self.bn1(y)  # [B, X, N, M]
        y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1))  # [B, X, 1, 1]
        y = y.view(y.size(0), -1)  # [B, X]
        return self.fc1(y)  # [B, _num_classes]


class ReductionCell(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.conv_reduce = nn.Conv2d(
            input_channels, output_channels, kernel_size=5, padding=2, stride=2
        )
        self.act = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(output_channels)

    def forward(self, input):
        return self.batch_norm(self.act(self.conv_reduce(input)))


def test_development():
    """

    :return:
    """

    """
        Arrange
    """
    # Define a customized cell constructor
    # Each cell has to map [batch_size, in_degree, a, b] -> [batch_size, 1, x, y]
    # Except for input cells, they map [batch_size, input_channel_size, a, b] -> [batch_size, 1, x, y]
    def my_cell_constructor(
        is_input, is_output, in_degree, out_degree, layer, input_channel_size
    ):
        if is_input:
            return ReductionCell(input_channel_size, 1)
        else:
            return ReductionCell(in_degree, 1)

    # Generate a random directed acyclic network
    # random_graph = nx.navigable_small_world_graph(200, 4, 5, 2)
    random_graph = nx.watts_strogatz_graph(200, 3, 0.8)
    adj_matrix = nx.convert_matrix.to_numpy_array(random_graph)
    directed_graph = nx.convert_matrix.from_numpy_array(np.tril(adj_matrix))

    # Pass the random network to cached layered graph as a structural wrapper
    structure = deepstruct.graph.CachedLayeredGraph()
    structure.add_nodes_from(directed_graph.nodes)
    structure.add_edges_from(directed_graph.edges)

    batch_size = 100
    input_channels = 3
    output_classes = 10
    model = deepstruct.sparse.DeepCellDAN(
        output_classes, input_channels, my_cell_constructor, structure
    )

    def count_parameters(model: torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    random_input = torch.tensor(
        np.random.randn(batch_size, input_channels, 50, 50),
        dtype=torch.float32,
        device=device,
    )

    # Act
    output = model(random_input)

    # Assert
    assert output.shape[0] == batch_size
    assert output.shape[1] == output_classes
    assert count_parameters(model) > 1
