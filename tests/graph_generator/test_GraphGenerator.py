import torch
import torch.nn.functional
# from draw_graph import draw_graph
from deepstruct.graph_generator import GraphGenerator
from tests.graph_generator.test_Architectures import *
from tests.graph_generator.test_TensorsData import TestTensorsData
from torch.nn.parameter import Parameter

graph_obj = GraphGenerator()


def weights_init(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape == torch.zeros([3, 6]).shape:
            m.weight = Parameter(TestTensorsData.get_input_tensor_weight_6_3())
    if isinstance(m, nn.Conv2d):
        print(m.weight.shape)
        if m.weight.shape == torch.zeros([3, 1, 2, 2]).shape:
            m.weight = Parameter(TestTensorsData.get_conv_weights_1_3())
            m.bias = Parameter(TestTensorsData.get_bias_3())
        elif m.weight.shape == torch.zeros([2, 1, 3, 3]).shape:
            m.weight = Parameter(TestTensorsData.get_conv_weights_1_2())
            m.bias = Parameter(TestTensorsData.get_bias_2())
        elif m.weight.shape == torch.zeros([1, 2, 3, 3]).shape:
            m.weight = Parameter(TestTensorsData.get_conv_weights_2_1())
            m.bias = Parameter(TestTensorsData.get_bias_1())


def test_max_pool_layer():
    # Arrange
    model = MaxPoolLayers()
    input_tensor = TestTensorsData.get_max_pool_input_tensor1()

    # Act
    directed_graph = graph_obj.generate_graph(model, 0.0, input_tensor)
    # # Assert
    assert directed_graph.number_of_edges() == 36
    assert directed_graph.number_of_nodes() == 45



def test_conv_max_pool_layer():
    # Arrange
    model = ConvMaxPoolLayers()
    model.apply(weights_init)
    input_tensor = TestTensorsData.get_conv_max_pool_input_tensor()

    # Act
    directed_graph = graph_obj.generate_graph(model, 0.0, input_tensor)
    # Assert
    assert directed_graph.number_of_edges() == 192


def test_conv_padding_layer():
    # Arrange
    model = Conv_Padding_Layers()
    model.apply(weights_init)

    input_tensor = TestTensorsData.get_conv_input_tensor()
    # Act
    directed_graph = graph_obj.generate_graph(model, 0.0, input_tensor)
    assert directed_graph.number_of_edges() == 162
    assert directed_graph.number_of_nodes() == 67


def test_conv_padding_layer_multi_in():
    # Arrange
    model = Conv_Padding_Multi_In()
    model.apply(weights_init)

    input_tensor = TestTensorsData.get_conv_input_tensor_2_1()
    # Act
    directed_graph = graph_obj.generate_graph(model, 0.0, input_tensor)
    assert directed_graph.number_of_edges() == 153
    assert directed_graph.number_of_nodes() == 106


def test_linear_layer():
    # Arrange
    model = LinearLayer()
    model.apply(weights_init)
    input_tensor = TestTensorsData.get_Linear_input_tensor()

    # Act
    directed_graph = graph_obj.generate_graph(model, 0.1, input_tensor.view(6, 1))
    # Assert
    assert directed_graph.number_of_edges() == 5

    # Act
    directed_graph = graph_obj.generate_graph(model, 0.0, input_tensor.view(6, 1))
    # Assert
    assert directed_graph.number_of_edges() == 8


test_max_pool_layer()
test_conv_max_pool_layer()
test_conv_padding_layer()
test_conv_padding_layer_multi_in()
test_linear_layer()
