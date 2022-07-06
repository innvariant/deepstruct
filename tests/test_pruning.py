import torch

import deepstruct.sparse

from deepstruct.pruning import PruningStrategy
from deepstruct.pruning import prune_layer_by_saliency
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
    for use_strategy, theta in [
        (PruningStrategy.PERCENTAGE, 50),
        (PruningStrategy.ABSOLUTE, 1000),
        (PruningStrategy.BUCKET, 100),
    ]:
        # Arrange
        model = deepstruct.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])
        set_random_saliency(model)

        # Act
        prune_network_by_saliency(model, theta, strategy=use_strategy)

        # Assert
        for ix, layer in enumerate(model.maskable_children):
            assert not torch.all(
                torch.eq(layer.mask, torch.ones(layer.weight.shape, dtype=torch.bool))
            ), f"Issue with {use_strategy} in layer {ix} - nothing was pruned."


def test_prune_layer_by_random_saliency():
    for use_strategy, theta in [
        (PruningStrategy.PERCENTAGE, 50),
        (PruningStrategy.ABSOLUTE, 1000),
    ]:
        # Arrange
        model = deepstruct.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])
        set_random_saliency(model)

        for ix, layer in enumerate(model.maskable_children):
            assert torch.all(
                torch.eq(layer.mask, torch.ones(layer.weight.shape, dtype=torch.bool))
            ), f"Mask should be initially one but was not in layer {ix}!"

        # Act
        prune_layer_by_saliency(model, theta, strategy=use_strategy)

        # Assert
        for ix, layer in enumerate(model.maskable_children):
            assert not torch.all(
                torch.eq(layer.mask, torch.ones(layer.weight.shape, dtype=torch.bool))
            ), f"Issue with {use_strategy} in layer {ix} - nothing was pruned."
