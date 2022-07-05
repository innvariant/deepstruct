import numpy as np
import torch

import deepstruct.pruning
import deepstruct.sparse
import deepstruct.util


def test_param_determines_mask_type():
    # Arrange
    layer1 = deepstruct.sparse.MaskedLinearLayer(5, 3, mask_as_params=False)
    layer2 = deepstruct.sparse.MaskedLinearLayer(5, 3, mask_as_params=True)

    # Assert
    assert layer1.mask.dtype == torch.bool
    assert layer2.mask.dtype == torch.int64


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
    # print(layer.mask)
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


def test_learn():
    # Arrange
    input_size = 5
    output_size = 2
    hidden_size = 10
    layer_one = deepstruct.sparse.MaskedLinearLayer(
        input_size, hidden_size, mask_as_params=True
    )
    layer_h2h = deepstruct.sparse.MaskedLinearLayer(
        hidden_size, hidden_size, mask_as_params=True
    )
    layer_out = deepstruct.sparse.MaskedLinearLayer(
        hidden_size, output_size, mask_as_params=True
    )
    layer_one.apply(deepstruct.pruning.set_random_masks)
    # layer_h2h.mask = torch.ones((hidden_size, hidden_size))
    for n_source in range(hidden_size):
        for n_target in range(hidden_size):
            layer_h2h[n_source, n_target] = (
                0 if n_source < hidden_size / 2 or n_target < hidden_size / 2 else 1
            )
    print(layer_h2h.mask)

    layer_out.apply(deepstruct.pruning.set_random_masks)

    samples_per_class = 5000
    ys = torch.cat([torch.ones(samples_per_class), torch.zeros(samples_per_class)])
    means = (ys * 2) - 1 + torch.randn_like(ys)
    input = torch.stack([torch.normal(means, 1) for _ in range(input_size)], dim=1)
    shuffling = torch.randperm(2 * samples_per_class)
    prop_train = 0.8
    offset_train = int(prop_train * len(shuffling))
    ids_train = shuffling[:offset_train]
    ids_test = shuffling[offset_train:]
    input_train = input[ids_train, :]
    input_test = input[ids_test, :]
    target_train = ys[ids_train].long()
    target_test = ys[ids_test].long()

    optimizer = torch.optim.Adam(
        list(layer_one.parameters())
        + list(layer_h2h.parameters())
        + list(layer_out.parameters()),
        lr=0.02,
        weight_decay=0.1,
    )
    loss = torch.nn.CrossEntropyLoss()

    # Act
    print(layer_one.weight)
    print(layer_h2h.weight)
    for _ in range(100):
        optimizer.zero_grad()
        h = layer_one(input_train)
        h = layer_h2h(torch.tanh(h))
        prediction = layer_out(torch.tanh(h))
        error = loss(prediction, target_train)
        error.backward()
        optimizer.step()
    print(layer_one.weight)
    print(layer_h2h.weight)

    # print(torch.round(torch.where(layer_h2h.mask.bool(), layer_h2h.weight*10**3, torch.zeros_like(layer_h2h.weight))) / (10**3))

    h = layer_one(input_test)
    h = layer_h2h(torch.tanh(h))
    prediction = layer_out(torch.tanh(h))

    accuracy = float(torch.sum(torch.argmax(prediction, axis=1) == target_test)) / len(
        target_test
    )
    assert accuracy > 0.5
