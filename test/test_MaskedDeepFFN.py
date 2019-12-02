import torch
import torch.utils
import unittest
import pypaddle.sparse
import numpy as np


class MaskedLinearLayerTest(unittest.TestCase):
    def test_get_structure(self):
        model = pypaddle.sparse.MaskedDeepFFN(784, 10, [20, 15, 12])
        structure = model.generate_structure(include_input=True, include_output=True)
        print(structure)
        # TODO

    def test_generate_large_structure(self):
        layers = [1000, 500, 500, 200, 100]
        model = pypaddle.sparse.MaskedDeepFFN((1, 28, 28), 10, layers)
        structure = model.generate_structure()

        self.assertEqual(len(layers), structure.num_layers)
        structure_layer_sizes = [structure.get_layer_size(l) for l in structure.layers]
        for l1, l2 in zip(structure_layer_sizes, layers):
            self.assertEqual(l1, l2, "Structure %s did not match definition %s" % (structure_layer_sizes, layers))

    def test_random_forward_possibly_on_gpu_success(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Arrange
        batch_size = 10
        input_size = 784
        output_size = 10
        model = pypaddle.sparse.MaskedDeepFFN(input_size, output_size, [200, 100, 50])
        model.to(device)
        random_input = torch.tensor(np.random.random((batch_size, input_size)), device=device, requires_grad=False)

        # Act
        output = model(random_input)

        # Assert
        self.assertEqual(output.numel(), batch_size*output_size)

    def test_random_forward_with_multiple_dimensions_success(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Arrange
        batch_size = 10
        input_size = (10, 5, 8)
        output_size = 10
        model = pypaddle.sparse.MaskedDeepFFN(input_size, output_size, [100, 200, 50])
        model.to(device)
        random_input = torch.tensor(np.random.random((batch_size,)+input_size), device=device, requires_grad=False)

        # Act
        output = model(random_input)

        # Assert
        self.assertEqual(output.numel(), batch_size*output_size)