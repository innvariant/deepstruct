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

    """import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(result, prog='dot')
    nx.draw(result, pos, with_labels=True, arrows=True)
    plt.show()"""


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

    """import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(result, prog='dot')
    nx.draw(result, pos, with_labels=True, arrows=True)
    plt.show()"""
