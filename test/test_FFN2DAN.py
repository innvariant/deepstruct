import itertools

import torch
import torch.utils
import unittest
import pypaddle.sparse
import numpy as np
import shutil

from pypaddle.util import get_mnist_loaders
from pypaddle.learning import train, test


class FFN2DANTest(unittest.TestCase):
    def setUp(self):
        self.possible_dataset_roots = ['/media/data/set/mnist', 'data/set/mnist']
        self.batch_size = 100
        self.train_loader, self.test_loader, _, self.dataset_root = get_mnist_loaders(self.batch_size, self.possible_dataset_roots)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        feature, labels = iter(self.train_loader).next()
        self.input_shape = feature.shape[1:]
        self.output_size = int(labels.shape[-1])

    def tearDown(self):
        if self.dataset_root is not self.possible_dataset_roots[0]:
            print('Deleting', self.dataset_root)
            shutil.rmtree(self.dataset_root)

    def test_transfer_random_reconnected_structure(self):
        loss = torch.nn.CrossEntropyLoss()

        num_epochs = 2
        model = pypaddle.sparse.MaskedDeepFFN(self.input_shape, self.output_size, [100, 50, 20])
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            train(self.train_loader, model, optimizer, loss, self.device)

        self.assertGreater(test(self.test_loader, model, self.device), 1/self.output_size)

        model.apply_mask()
        model.recompute_mask(theta=0.01)
        structure = model.generate_structure(include_input=False, include_output=False)

        for source_layer in structure.layers:
            source_layer_size = structure.get_layer_size(source_layer)
            for target_layer in structure.layers[source_layer+1:]:
                target_layer_size = structure.get_layer_size(target_layer)

                random_source_nodes = np.random.choice(
                    structure.get_vertices(source_layer),
                    np.random.randint(1, source_layer_size+1),
                    replace=False
                )
                random_target_nodes = np.random.choice(
                    structure.get_vertices(target_layer),
                    np.random.randint(1, target_layer_size+1),
                    replace=False
                )

                structure.add_edges_from([edge for edge in itertools.product(random_source_nodes, random_target_nodes)])

        dan_model = pypaddle.sparse.MaskedDeepDAN(self.input_shape, self.output_size, structure)
        dan_model.to(self.device)
        dan_optimizer = torch.optim.SGD(dan_model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            train(self.train_loader, dan_model, dan_optimizer, loss, self.device)

        self.assertGreater(test(self.test_loader, dan_model, self.device), 1/self.output_size)
