from abc import abstractmethod
from typing import Callable

import torch.nn


class NodeMapper:

    @abstractmethod
    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class CustomNodeMap:

    def __init__(self, node_mappers: dict, default_mapper: NodeMapper):
        self.node_mappers = node_mappers
        self.default_mapper = default_mapper

    def map_node(self, func_module, graph, output_shape, func_name, input_shape=None):
        # node_name = func_module.__name__ + " " + func_name
        node_name = func_name
        self.node_mappers.get(func_module, self.default_mapper).add_node(graph, output_shape, node_name, input_shape)


class HighLevelNodeMap(CustomNodeMap):

    def __init__(self):
        super().__init__({}, All2VertexNodeMapper())


class LowLevelNodeMap(CustomNodeMap):

    def __int__(self):
        super().__init__({
            torch.nn.Linear: Linear2LayerMapper(),

        }, All2VertexNodeMapper())


class All2VertexNodeMapper(NodeMapper):

    def __init__(self):
        self.layer = 0

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        added_node = graph.add_vertex(layer=self.layer, name=func_name)
        if self.layer > 0:
            graph.add_edge(self.layer - 1, added_node)
        self.layer += 1


class Linear2LayerMapper(NodeMapper):

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class Linear2VertexMapper(NodeMapper):
    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class Conv2LayerMapper(NodeMapper):

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class Conv2VertexMapper(NodeMapper):

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class GenericVertexMapper(NodeMapper):

    def __int__(self, add_node_implementation: Callable):
        self.add_node_implementation = add_node_implementation

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        self.add_node_implementation(graph, output_shape, func_name, input_shape)
