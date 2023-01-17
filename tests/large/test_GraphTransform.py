import time

import networkx as nx
import torch
import torchvision

from torchvision import datasets
from torchvision import transforms

import deepstruct.graph
import deepstruct.sparse

from deepstruct.transform import GraphTransform


path_common_mnist = "/media/data/set/mnist/"


def test_mnist_large():
    learning_rate = 0.001
    batch_size = 100

    random_graph = nx.newman_watts_strogatz_graph(100, 4, 0.5)
    structure = deepstruct.graph.CachedLayeredGraph()
    structure.add_edges_from(random_graph.edges)
    structure.add_nodes_from(random_graph.nodes)

    # Build a neural network classifier with 784 input and 10 output neurons and the given structure
    model = deepstruct.sparse.MaskedDeepDAN(784, 10, structure)
    model.apply_mask()  # Apply the mask on the weights (hard, not undoable)
    model.recompute_mask()  # Use weight magnitude to recompute the mask from the network
    pruned_structure = (
        model.generate_structure()
    )  # Get the structure -- a networkx graph -- based on the current mask

    new_model = deepstruct.sparse.MaskedDeepDAN(784, 10, pruned_structure)

    # Define transform to normalize data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Download and load the training data
    train_set = datasets.MNIST(
        path_common_mnist, download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    test_set = datasets.MNIST(
        path_common_mnist, download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(new_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for feat, target in trainloader:
        optimizer.zero_grad()
        prediction = new_model(feat)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

    new_model.eval()
    for feat, target in testloader:
        prediction = new_model(feat)
        loss = criterion(prediction, target)
        print(loss)


def test_torchvision_large():
    shape_input = (3, 224, 224)
    print("Loading model")
    model = torchvision.models.alexnet(pretrained=True)
    print("Model loaded")

    functor = GraphTransform(torch.randn((1,) + shape_input))

    # Act
    print("Start functor transformation")
    result = functor.transform(model)
    print("Functor transformation done")

    print(len(result.nodes))
    print(len(result.edges))


def test_generate_structure_on_large_maskeddeepffn_success():
    # Arrange
    shape_input = (1, 28, 28)
    layers = [1000, 500, 500, 200, 100]
    model = deepstruct.sparse.MaskedDeepFFN(shape_input, 10, layers)

    functor = GraphTransform(torch.randn((1,) + shape_input))

    # Act
    time_transform_start = time.time()
    structure = functor.transform(model)
    time_transform_end = time.time()
    print(
        f"Took {round(time_transform_end-time_transform_start, 4)} to transform large structure."
    )

    # Assert
    assert 2 + len(layers) == structure.num_layers
    structure_layer_sizes = [
        structure.get_layer_size(lay) for lay in structure.layers[1:-1]
    ]
    for ix, (l1, l2) in enumerate(zip(structure_layer_sizes, layers)):
        assert (
            l1 == l2
        ), f"Structure {structure_layer_sizes} did not match definition {layers} as layer {ix}"
