import torch

from deepstruct.transform import GraphTransform


def dev_test_stacked_graph():
    # Arrange
    # a linear module with larger input than its output
    size_input = 10
    size_hidden = 5
    model = torch.nn.Sequential(
        torch.nn.ModuleList(
            [torch.nn.Linear(size_input, size_hidden), torch.nn.Conv2d(1, 1, (3, 3))]
        )
    )
    next(model.children())[0].weight[
        :, :
    ] += 1  # Make sure each weight is large enough so none is getting "pruned"

    input_random = torch.randn(size_input)
    output_random = model(input_random)
    print(output_random)

    functor = GraphTransform(size_input)

    # Act
    result = functor.transform(model)
    print(result.nodes)
