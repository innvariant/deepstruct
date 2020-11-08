import itertools

import numpy as np
import torch

from deepstruct.graph import LabeledDAG
from deepstruct.transform import LinearLayerFunctor


def test_linear_new_independent_graph():
    size_layer_1 = 10
    size_layer_2 = 5
    model = torch.nn.Linear(size_layer_1, size_layer_2)
    model.weight[
        :, :
    ] += 1  # Make sure each weight is large enough so none is getting "pruned"

    functor = LinearLayerFunctor(threshold=0.01)

    result = functor.transform(model)

    assert len(result.nodes) == size_layer_1 + size_layer_2
    assert len(result.edges) == size_layer_1 * size_layer_2
    assert result.num_layers == 2


def test_linear_simple_with_prior_graph():
    prior_graph = LabeledDAG()
    size_layer_1 = 4
    size_layer_2 = 4
    prior_graph.add_edges_from(
        itertools.product(
            np.arange(size_layer_1),
            np.arange(size_layer_1 + 1, size_layer_1 + size_layer_2 + 1),
        )
    )

    model = torch.nn.Linear(10, 5)

    functor = LinearLayerFunctor(graph=prior_graph, threshold=0.01)

    functor.transform(model)
