import numpy as np
import torch

from deepstruct.transform import Conv2dLayerFunctor


def no_test_conv_simple():
    # Arrange
    input_width = 5
    input_height = 5
    channels_in = 3
    kernel_size = (3, 3)
    model = torch.nn.Conv2d(
        in_channels=channels_in, out_channels=2, kernel_size=kernel_size, stride=1
    )
    model.weight[
        :, :
    ] += 10  # Make sure each weight is large enough so none is getting "pruned"

    functor = Conv2dLayerFunctor(input_width, input_height, threshold=0.01)

    # Act
    result = functor.transform(model)
    output = model.forward(torch.rand(size=(1, channels_in, input_height, input_width)))
    number_output_features = np.prod(output.shape)

    # Assert
    assert result.last_layer_size == number_output_features
    print(result.first_layer_size)
    print(result.last_layer_size)
    print(len(result.edges))
    print(
        [
            v
            for v in result.nodes
            if result.get_layer(v) != 0 and result.in_degree(v) == 0
        ]
    )
    print(
        [
            v
            for v in result.nodes
            if result.get_layer(v) != 1 and result.out_degree(v) == 0
        ]
    )
    # TODO: check connectivity


def no_test_conv_nonsquare_kernel():
    # Arrange
    input_width = 20
    input_height = 10
    channels_in = 3
    kernel_size = (6, 3)
    assert kernel_size[0] != kernel_size[1]
    model = torch.nn.Conv2d(
        in_channels=channels_in, out_channels=2, kernel_size=kernel_size, stride=1
    )
    model.weight[
        :, :
    ] += 10  # Make sure each weight is large enough so none is getting "pruned"

    functor = Conv2dLayerFunctor(input_width, input_height, threshold=0.01)

    # Act
    result = functor.transform(model)
    output = model.forward(torch.rand(size=(1, channels_in, input_height, input_width)))
    number_output_features = np.prod(output.shape)

    # Assert
    assert result.last_layer_size == number_output_features


def no_test_conv_multiple_configs():
    # Arrange
    input_width = 10
    input_height = 10
    models = []
    channels_in = 3

    for stride in range(1, 3):
        for channels_out in range(1, 5):
            for kernel_dim_size in range(2, 7):
                kernel_size = (kernel_dim_size, kernel_dim_size)
                model = torch.nn.Conv2d(
                    in_channels=channels_in,
                    out_channels=channels_out,
                    kernel_size=kernel_size,
                    stride=stride,
                )
                model.weight[
                    :, :
                ] += 10  # Make sure each weight is large enough so none is getting "pruned"
                models.append(model)

    functor = Conv2dLayerFunctor(input_width, input_height, threshold=0.01)

    # Act
    for model in models:
        result = functor.transform(model)
        output = model.forward(
            torch.rand(size=(1, channels_in, input_height, input_width))
        )
        number_output_features = np.prod(output.shape)

        # Assert
        assert result.last_layer_size == number_output_features
