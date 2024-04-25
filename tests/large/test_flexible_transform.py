import torch
import torchvision.models

from deepstruct.flexible_transform import GraphTransform
from deepstruct.traverse_strategies import FXTraversal
from tests.utils import plot_graph


def test_fx_resnet18():
    print(" ")
    resnet = torchvision.models.resnet18(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(input_tensor, traversal_strategy=FXTraversal())
    graph_transformer.transform(resnet)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation resnet")


def test_fx_alexnet():
    print(" ")
    alexnet = torchvision.models.alexnet(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(
        input_tensor,
        traversal_strategy=FXTraversal(
            unfold_modules=[object],
            fold_modules=[
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.BatchNorm2d,
                torch.nn.MaxPool2d,
            ],
        ),
    )
    graph_transformer.transform(alexnet)
    graph = graph_transformer.get_graph()
    plot_graph(graph, "Transformation alexnet")


def test_fx_resnet50():
    print(" ")
    res50 = torchvision.models.resnet50(True)
    input_tensor = torch.rand(1, 3, 224, 224)
    graph_transformer = GraphTransform(
        input_tensor, traversal_strategy=FXTraversal(exclude_fn=[torch.add])
    )
    graph_transformer.transform(res50)
    plot_graph(graph_transformer.get_graph(), "Transformation resnet 50")
