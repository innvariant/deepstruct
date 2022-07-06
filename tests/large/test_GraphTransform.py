import time

import torch
import torchvision

import deepstruct.sparse

from deepstruct.transform import GraphTransform


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
