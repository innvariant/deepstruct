import numpy as np
import torch
import torch.utils

import deepstruct.sparse


def test_set_random_masks():
    model = deepstruct.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])

    for layer in deepstruct.sparse.maskable_layers(model):
        random_mask = torch.tensor(np.random.binomial(1, 0.5, layer.mask.shape))
        layer.mask = random_mask


def test_prune():
    model = deepstruct.sparse.MaskedDeepFFN(784, 10, [1000, 500, 200, 100])
    model.recompute_mask(theta=0.01)
    model.apply_mask()

    for layer in deepstruct.sparse.maskable_layers(model):
        print(layer.mask.shape)
        print(torch.sum(layer.mask) / float(torch.numel(layer.mask)))


def test_random_forward_possibly_on_gpu_success():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    batch_size = 10
    input_size = 784
    output_size = 10
    model = deepstruct.sparse.MaskedDeepFFN(input_size, output_size, [200, 100, 50])
    model.to(device)
    random_input = torch.tensor(
        np.random.random((batch_size, input_size)), device=device, requires_grad=False
    )

    # Act
    output = model(random_input)

    # Assert
    assert output.numel() == batch_size * output_size


def test_random_forward_with_multiple_dimensions_success():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Arrange
    batch_size = 10
    input_size = (10, 5, 8)
    output_size = 10
    model = deepstruct.sparse.MaskedDeepFFN(input_size, output_size, [100, 200, 50])
    model.to(device)
    random_input = torch.tensor(
        np.random.random((batch_size,) + input_size), device=device, requires_grad=False
    )

    # Act
    output = model(random_input)

    # Assert
    assert output.numel() == batch_size * output_size
