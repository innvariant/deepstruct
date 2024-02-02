import torch
import torch.nn.functional as F

import deepstruct.pruning
import deepstruct.sparse


def test_learn():
    # Arrange
    input_size = 5
    output_size = 2
    hidden_size = 10
    layer_one = deepstruct.sparse.MaskedLinearLayer(
        input_size, hidden_size, mask_as_params=False
    )
    layer_h2h = deepstruct.sparse.MaskedLinearLayer(
        hidden_size, hidden_size, mask_as_params=False
    )
    layer_out = deepstruct.sparse.MaskedLinearLayer(
        hidden_size, output_size, mask_as_params=False
    )
    layer_one.apply(deepstruct.pruning.set_random_masks)
    # layer_h2h.mask = torch.ones((hidden_size, hidden_size))
    for n_source in range(hidden_size):
        for n_target in range(hidden_size):
            layer_h2h[n_source, n_target] = (
                0 if n_source < hidden_size / 2 or n_target < hidden_size / 2 else 1
            )
    print(layer_h2h.weight)
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
    h2h_priortraining = torch.clone(layer_h2h.weight)
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
    h2h_posttraining = torch.clone(layer_h2h.weight)

    print("Diffs h2h weights")
    print(torch.abs(h2h_priortraining - h2h_posttraining))

    # print(torch.round(torch.where(layer_h2h.mask.bool(), layer_h2h.weight*10**3, torch.zeros_like(layer_h2h.weight))) / (10**3))

    h = layer_one(input_test)
    print("h", h)
    h_nomask = F.linear(torch.tanh(h), layer_h2h.weight, layer_h2h.bias)
    print("h_nomask", h_nomask)
    h = layer_h2h(torch.tanh(h))
    print("h", h)
    prediction = layer_out(torch.tanh(h))
    print("h2h_bias", layer_h2h.bias)

    accuracy = float(torch.sum(torch.argmax(prediction, axis=1) == target_test)) / len(
        target_test
    )
    assert accuracy > 0.5
