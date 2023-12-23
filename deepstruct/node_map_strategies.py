import itertools
import string
from abc import abstractmethod
from typing import Callable

import networkx
import numpy as np
import py
import torch.nn

from deepstruct.graph import LabeledDAG


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
        node_name = kwargs.get('name')
        if name_indices.get(node_name, 0) is 0:
            name_indices[node_name] = []
        name_indices[node_name].append(index)
        for p_node in predecessors:
            pn_index = name_indices.get(str(p_node), "not found")
            if pn_index is "not found" or len(pn_index) == 0:
                if str(p_node) in ignored_names:
                    nodes = graph.get_vertices(graph.current_layer - 1)
                    for node in nodes:
                        graph.add_edge(node, index)
                else:
                    continue
            else:
                for e in pn_index:
                    graph.add_edge(e, index)


class Linear2LayerMapper(NodeMapper):

    def __init__(self):
        self.name_index = {}

    def add_node(self, graph, predecessors, **kwargs):
        model = kwargs.get("origin_module")
        sources = []
        for _ in range(model.in_features):
            sources.append(self._add_node(graph, predecessors, **kwargs))

    def _add_node(self, graph, predecessors, **kwargs):
        name_indices = kwargs.get('name_indices')
        ignored_names = kwargs.get('ignored')
        index = graph.add_vertex(graph.current_layer,
                                 name=kwargs.get('name'),
                                 shape=kwargs.get('shape'),
                                 module=kwargs.get('module')
                                 )
        node_name = kwargs.get('name')
        if name_indices.get(node_name, 0) is 0:
            name_indices[node_name] = []
        name_indices[node_name].append(index)
        for p_node in predecessors:
            pn_index = name_indices.get(str(p_node), "not found")
            print(p_node, pn_index)
            if pn_index is "not found" or len(pn_index) == 0:
                if str(p_node) in ignored_names:
                    nodes = graph.get_vertices(graph.current_layer - 1)
                    for node in nodes:
                        graph.add_edge(node, index)
                else:
                    continue
            else:
                for e in pn_index:
                    graph.add_edge(e, index)
        return index

