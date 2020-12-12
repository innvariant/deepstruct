import numpy as np
import torch

import deepstruct.pruning
import deepstruct.sparse
import deepstruct.util


def test_set_mask_explicitly_success():
    input_size = 5
    output_size = 2
    layer = deepstruct.sparse.MaskedLinearLayer(input_size, output_size)
    mask = torch.zeros((output_size, input_size), dtype=torch.bool)
    mask[0, 0] = 1
    mask[0, 1] = 1
    mask[1, 2] = 1

    layer.mask = mask

    assert np.all(np.equal(np.array(mask), np.array(layer.mask)))


def test_parameter_reset_success():
    # Arrange - initialize a masked layer and randomize its mask
    input_size = 5
    output_size = 7
    layer = deepstruct.sparse.MaskedLinearLayer(input_size, output_size)
    layer.apply(deepstruct.pruning.set_random_masks)
    initial_state = np.copy(layer.mask)

    # Act - Now the mask should be reset to only ones
    layer.reset_parameters()

    # Assert - The random mask and the resetted mask should not match
    assert layer.mask.size() == initial_state.shape
    assert (np.array(layer.mask) != initial_state).any()


def test_mask_changes_output_success():
    input_size = 5
    output_size = 7
    layer = deepstruct.sparse.MaskedLinearLayer(input_size, output_size)
    input = torch.rand(input_size)

    layer.apply(deepstruct.pruning.set_random_masks)
    first_mask = np.copy(layer.mask)
    first_mask_output = layer(input).detach().numpy()
    layer.apply(deepstruct.pruning.set_random_masks)
    second_mask = np.copy(layer.mask)
    second_mask_output = layer(input).detach().numpy()

    assert (
        first_mask != second_mask
    ).any(), "Masks for inference should not equal, but are randomly generated."
    assert np.any(np.not_equal(first_mask_output, second_mask_output))


def test_random_input_success():
    input_size = 5
    output_size = 2
    model = deepstruct.sparse.MaskedLinearLayer(input_size, output_size)
    input = torch.tensor(np.random.random(input_size))

    output = model(input)

    assert output.numel() == output_size


def test_initialize_random_parameterizable_mask_success():
    # Arrange - initialize a masked layer and randomize its mask
    input_size = 20
    output_size = 10
    layer = deepstruct.sparse.MaskedLinearLayer(
        input_size, output_size, mask_as_params=True
    )
    initial_state = np.copy(layer.mask)

    # Act
    layer.apply(deepstruct.pruning.set_random_masks)

    # Assert
    assert (np.array(layer.mask) != initial_state).any()


def test_paramterized_masks_contained_in_model_params():
    # Arrange - initialize a masked layer and randomize its mask
    name_param = "_mask"
    input_size = 5
    output_size = 7
    layer = deepstruct.sparse.MaskedLinearLayer(
        input_size, output_size, mask_as_params=True
    )

    params = {name: p for name, p in layer.named_parameters()}

    assert len(list(layer.parameters())) == 3
    assert name_param in params
    assert params[name_param].numel() == 2 * input_size * output_size


def test_nonparamterized_masks_not_contained_in_model_params():
    # Arrange - initialize a masked layer and randomize its mask
    name_param = "_mask"
    input_size = 5
    output_size = 7
    layer = deepstruct.sparse.MaskedLinearLayer(
        input_size, output_size, mask_as_params=False
    )

    params = {name: p for name, p in layer.named_parameters()}

    assert name_param not in params
    assert len(list(layer.parameters())) == 2


def test_paramterized_masks_success():
    # Arrange - initialize a masked layer and randomize its mask
    input_size = 5
    output_size = 7
    layer = deepstruct.sparse.MaskedLinearLayer(
        input_size, output_size, mask_as_params=True
    )
    # layer.apply(deepstruct.pruning.set_random_masks)
    print(layer.mask)
    initial_alpha_mask = layer._mask.clone().detach().cpu().numpy()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1, weight_decay=0.1)

    # Act
    for _ in range(10):
        optimizer.zero_grad()
        loss = torch.sum(torch.abs(layer._mask[:, :, 1]))
        loss.backward()
        optimizer.step()

    # Assert<
    assert layer._mask.size() == initial_alpha_mask.shape
    assert (layer._mask.clone().detach().cpu().numpy() != initial_alpha_mask).any()
    print(layer._mask)
    print(layer.mask)
