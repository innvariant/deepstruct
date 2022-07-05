import torch

import deepstruct.sparse

from deepstruct.pruning import PruningStrategy
from deepstruct.pruning import prune_network_by_saliency
from deepstruct.pruning import set_random_saliency


def test_set_random_saliency():
    # Arrange
    model = deepstruct.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])

    # Act
    set_random_saliency(model)

    # Assert
    for layer in model.maskable_children:
        assert not torch.equal(layer.saliency, layer.weight)


def test_prune_network_by_random_saliency():
    # Arrange
    model = deepstruct.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])
    set_random_saliency(model)

    # Act
    prune_network_by_saliency(model, 0.9, strategy=PruningStrategy.PERCENTAGE)

    # Assert
    for ix, layer in enumerate(model.maskable_children):
        assert not torch.all(
            torch.eq(layer.mask, torch.ones(layer.weight.shape, dtype=torch.bool))
        ), f"Issue in layer {ix} - nothing was pruned."
