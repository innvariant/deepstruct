import networkx as nx
import torch

import deepstruct.sparse

from deepstruct.transform import GraphTransform


class SimpleNet(torch.nn.Module):
    def __init__(self, size_input: int, size_hidden: int):
        super().__init__()
        self._size_hidden = size_hidden
        self._linear1 = torch.nn.Linear(size_input, size_hidden * size_hidden)
        self._linear2 = torch.nn.Linear(
            size_hidden * size_hidden, size_hidden * size_hidden
        )
        self._conv = torch.nn.Conv2d(1, 1, (3, 3))
        self._linear3 = torch.nn.Linear(1 * 2 * 2, 10)
        self._act = torch.nn.ReLU()

    def forward(self, input):
        out = self._act(self._linear2(self._act(self._linear1(input))))
        out = out.reshape((1, 1, self._size_hidden, self._size_hidden))
        out = self._conv(out).flatten()
        return self._linear3(out)


def test_stacked_graph():
    # Arrange
    # a linear module with larger input than its output
    size_input = 20
    size_hidden = 4
    model = SimpleNet(size_input, size_hidden)
    model._linear1.weight[
        :, :
    ] += 1  # Make sure each weight is large enough so none is getting "pruned"
    model._linear2.weight[:, :] += 1
    model._conv.weight[:, :] += 1

    functor = GraphTransform(torch.randn((1, 20)))

    # Act
    result = functor.transform(model)
    print(result.nodes)


def test_deep_ffn():
    # Arrange
    # a linear module with larger input than its output
    shape_input = (50,)
    layers = [100, 50, 100]
    output_size = 10
    model = deepstruct.sparse.MaskedDeepFFN(shape_input, output_size, layers)
    for layer in deepstruct.sparse.maskable_layers(model):
        layer.weight[:, :] += 1  # make sure everything is fully connected

    functor = GraphTransform(torch.randn((1,) + shape_input))

    # Act
    result = functor.transform(model)

    assert len(result.nodes) == shape_input[0] + sum(layers) + output_size
    assert (
        len(result.edges)
        == shape_input[0] * layers[0]
        + sum(l1 * l2 for l1, l2 in zip(layers[0:-1], layers[1:]))
        + layers[-1] * output_size
    )


def test_deep_ffn2():
    # Arrange
    # a linear module with larger input than its output
    shape_input = (50,)
    layers = [100] * 100
    output_size = 10
    model = deepstruct.sparse.MaskedDeepFFN(shape_input, output_size, layers)
    for layer in deepstruct.sparse.maskable_layers(model):
        layer.weight[:, :] += 1  # make sure everything is fully connected

    functor = GraphTransform(torch.randn((1,) + shape_input))

    # Act
    result = functor.transform(model)

    assert len(result.nodes) == shape_input[0] + sum(layers) + output_size
    assert (
        len(result.edges)
        == shape_input[0] * layers[0]
        + sum(l1 * l2 for l1, l2 in zip(layers[0:-1], layers[1:]))
        + layers[-1] * output_size
    )


def test_deep_dan():
    # Arrange
    shape_input = (50,)
    output_size = 10
    random_graph = nx.newman_watts_strogatz_graph(100, 4, 0.5)
    print(len(random_graph.nodes))
    print(len(random_graph.edges))
    structure = deepstruct.graph.CachedLayeredGraph()
    structure.add_edges_from(random_graph.edges)
    structure.add_nodes_from(random_graph.nodes)
    model = deepstruct.sparse.MaskedDeepDAN(shape_input, output_size, structure)
    for layer in deepstruct.sparse.maskable_layers(model):
        layer.weight[:, :] += 1  # make sure everything is fully connected

    functor = GraphTransform(torch.randn((1,) + shape_input))

    # Act
    result = functor.transform(model)

    print(len(result.nodes))
    print(len(result.edges))
