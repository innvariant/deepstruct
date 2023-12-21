from abc import abstractmethod
from typing import Callable

import networkx
import torch.nn


class NodeMapper:

    @abstractmethod
    def add_node(self, graph, predecessors, **kwargs):
        pass


class CustomNodeMap:

    def __init__(self, node_mappers: dict, default_mapper: NodeMapper):
        self.node_mappers = node_mappers
        self.default_mapper = default_mapper
        self.ignored_nodes = []
        self.name_indices = {}

    def map_node(self, graph, module, predecessors, **kwargs):
        kwargs['module'] = module
        kwargs['ignored'] = self.ignored_nodes
        kwargs['name_indices'] = self.name_indices
        self.node_mappers.get(module, self.default_mapper).add_node(graph, predecessors, **kwargs)

    def ignore_node(self, name):
        self.ignored_nodes.append(name)


class HighLevelNodeMap(CustomNodeMap):

    def __init__(self):
        super().__init__({}, All2VertexNodeMapper())


class LowLevelNodeMap(CustomNodeMap):

    def __init__(self):
        super().__init__({
            torch.nn.Linear: Linear2LayerMapper(),
        }, All2VertexNodeMapper())


class All2VertexNodeMapper(NodeMapper):

    def add_node(self, graph, predecessors, **kwargs):
        name_indices = kwargs.pop('name_indices')
        ignored_names = kwargs.pop('ignored')
        index = graph.add_vertex(graph.current_layer, **kwargs)
        name_indices[kwargs.get('name')] = index
        for p_node in predecessors:
            pn_index = name_indices.get(str(p_node), - 1)
            if pn_index != -1:
                graph.add_edge(pn_index, index)
            elif str(p_node) in ignored_names:
                nodes = graph.get_vertices(graph.current_layer - 1)
                for node in nodes:
                    graph.add_edge(node, index)


class Linear2LayerMapper(NodeMapper):

    def __init__(self):
        self.name_index = {}

    def add_node(self, graph, predecessors, **kwargs):
        print("Linear variant chosen for: ", kwargs.get('name'))
        module = kwargs.get("origin_module")
        print(module)
        name_indices = kwargs.pop('name_indices')
        ignored_names = kwargs.pop('ignored')
        index = graph.add_vertex(graph.current_layer, **kwargs)
        name_indices[kwargs.get('name')] = index
        for p_node in predecessors:
            pn_index = name_indices.get(str(p_node), - 1)
            if pn_index != -1:
                graph.add_edge(pn_index, index)
            elif str(p_node) in ignored_names:
                nodes = graph.get_vertices(graph.current_layer - 1)
                for node in nodes:
                    graph.add_edge(node, index)
