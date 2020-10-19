import numpy as np
import torch

import pypaddle.pruning
import pypaddle.sparse
import pypaddle.util


def test_set_mask_explicitly_success():
    input_size = 5
    output_size = 2
    layer = pypaddle.sparse.MaskedLinearLayer(input_size, output_size)
    mask = torch.zeros((output_size, input_size), dtype=torch.bool)
    mask[0, 0] = 1
    mask[0, 1] = 1
    mask[1, 2] = 1

    layer.set_mask(mask)

    assert np.all(np.equal(np.array(mask), np.array(layer.mask)))


def test_parameter_reset_success():
    # Arrange - initialize a masked layer and randomize its mask
    input_size = 5
    output_size = 7
    layer = pypaddle.sparse.MaskedLinearLayer(input_size, output_size)
    layer.apply(pypaddle.pruning.set_random_masks)
    initial_state = np.copy(layer.mask)

    # Act - Now the mask should be reset to only ones
    layer.reset_parameters()

    # Assert - The random mask and the resetted mask should not match
    assert layer.mask.size() == initial_state.shape
    assert (np.array(layer.mask) != initial_state).any()


def test_mask_changes_output_success():
    input_size = 5
    output_size = 7
    layer = pypaddle.sparse.MaskedLinearLayer(input_size, output_size)
    input = torch.rand(input_size)

    layer.apply(pypaddle.pruning.set_random_masks)
    first_mask = np.copy(layer.mask)
    first_mask_output = layer(input).detach().numpy()
    layer.apply(pypaddle.pruning.set_random_masks)
    second_mask = np.copy(layer.mask)
    second_mask_output = layer(input).detach().numpy()

    assert (
        first_mask != second_mask
    ).any(), "Masks for inference should not equal, but are randomly generated."
    assert np.any(np.not_equal(first_mask_output, second_mask_output))


def test_random_input_success():
    input_size = 5
    output_size = 2
    model = pypaddle.sparse.MaskedLinearLayer(input_size, output_size)
    input = torch.tensor(np.random.random(input_size))

    output = model(input)

    assert output.numel() == output_size
