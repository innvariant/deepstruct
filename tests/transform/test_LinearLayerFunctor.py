import numpy as np
import torch

from deepstruct.transform import LinearLayerFunctor


def test_linear_simple_compression():
    # Arrange
    # a linear module with larger input than its output
    size_layer_1 = 10
    size_layer_2 = 5
    assert size_layer_2 < size_layer_1
    model = torch.nn.Linear(size_layer_1, size_layer_2)
    # Make sure each weight is large enough so none is getting "pruned"
    model.weight.data += 1
    functor = LinearLayerFunctor(threshold=0.01)

    # Act
    result = functor.transform(model)

    # Assert
    assert len(result.nodes) == size_layer_1 + size_layer_2
    assert len(result.edges) == size_layer_1 * size_layer_2
    assert result.num_layers == 2


def test_linear_simple_expansion():
    # Arrange
    # a linear module with a smaller input than output
    size_layer_1 = 4
    size_layer_2 = 8
    assert size_layer_2 > size_layer_1
    model = torch.nn.Linear(size_layer_1, size_layer_2)
    model.weight.data = torch.tensor(
        np.random.uniform(1, 2, size=(size_layer_2, size_layer_1)), dtype=torch.float32
    )
    functor = LinearLayerFunctor(threshold=0.01)

    # Act
    result = functor.transform(model)

    # Assert
    assert len(result.nodes) == size_layer_1 + size_layer_2
    assert len(result.edges) == size_layer_1 * size_layer_2
    assert result.num_layers == 2


def test_linear_sparse():
    # Arrange
    # calculate a sparse binary matrix and create a linear module of it
    size_layer_1 = 5
    size_layer_2 = 10
    model = torch.nn.Linear(size_layer_1, size_layer_2)
    weights = np.random.uniform(1, 2, size=(size_layer_2, size_layer_1))
    mask = np.random.binomial(1, 0.3, size=(size_layer_2, size_layer_1))
    num_edges = np.sum(mask)
    model.weight.data = torch.tensor(weights * mask, dtype=torch.float32)
    functor = LinearLayerFunctor(threshold=0.01)

    # Act
    result = functor.transform(model)

    # Assert
    assert len(result.nodes) == size_layer_1 + size_layer_2
    assert len(result.edges) == num_edges
    assert result.num_layers == 2
