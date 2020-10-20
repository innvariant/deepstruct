import itertools

import numpy as np
import pypaddle.sparse
import torch
import torch.utils

from pypaddle.learning import run_evaluation
from pypaddle.learning import train

from tests.util import get_mnist_loaders


def test_transfer_random_reconnected_structure():
    batch_size = 100
    train_loader, test_loader, _, dataset_root = get_mnist_loaders(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature, labels = iter(train_loader).next()
    input_shape = feature.shape[1:]
    output_size = int(labels.shape[-1])

    loss = torch.nn.CrossEntropyLoss()

    num_epochs = 2
    model = pypaddle.sparse.MaskedDeepFFN(input_shape, output_size, [100, 50, 20])
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss, device)

    assert run_evaluation(test_loader, model, device) > 1 / output_size

    model.apply_mask()
    model.recompute_mask(theta=0.01)
    structure = model.generate_structure(include_input=False, include_output=False)

    for source_layer in structure.layers:
        source_layer_size = structure.get_layer_size(source_layer)
        for target_layer in structure.layers[source_layer + 1 :]:
            target_layer_size = structure.get_layer_size(target_layer)

            random_source_nodes = np.random.choice(
                structure.get_vertices(source_layer),
                np.random.randint(1, source_layer_size + 1),
                replace=False,
            )
            random_target_nodes = np.random.choice(
                structure.get_vertices(target_layer),
                np.random.randint(1, target_layer_size + 1),
                replace=False,
            )

            structure.add_edges_from(
                [
                    edge
                    for edge in itertools.product(
                        random_source_nodes, random_target_nodes
                    )
                ]
            )

    dan_model = pypaddle.sparse.MaskedDeepDAN(input_shape, output_size, structure)
    dan_model.to(device)
    dan_optimizer = torch.optim.SGD(dan_model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        train(train_loader, dan_model, dan_optimizer, loss, device)

    assert run_evaluation(test_loader, dan_model, device) > 1 / output_size
