from typing import List

import networkx


class LayeredGraph:

    def __init__(self):
        self.graph = networkx.DiGraph()
        self.vertex_id = 0
        self.layer_before = []

    def add_vertex(self, name):
        self.graph.add_node(self.vertex_id, name=name)
        if len(self.layer_before) > 0:
            for vertex in self.layer_before:
                self.graph.add_edge(vertex, self.vertex_id)
        self.layer_before = [self.vertex_id]
        self.vertex_id += 1

    def add_vertices(self, name, count):
        current_vertices = []
        for i in range(count):
            self.graph.add_node(self.vertex_id, name=name)
            current_vertices.append(self.vertex_id)
            self.vertex_id += 1
        if len(self.layer_before) > 0:
            for vertex in self.layer_before:
                for new_vertex in current_vertices:
                    self.graph.add_edge(vertex, new_vertex)
        self.layer_before = current_vertices

    def add_edge(self, src, dest):
        self.graph.add_edge(src, dest)


class LayeredFXGraph(networkx.DiGraph):

    def __init__(self, **attr):
        super(LayeredFXGraph, self).__init__(**attr)
        self._layers = {}
        self._index_name_map = {}
        self._name_index_map = {}
        self._node_edges_mask = {}
        self.ignored_nodes = []
        self.current_layer = 0

    def get_next_layer_index(self):
        return len(self._layers)

    def get_nodes_from_layer(self, layer_index):
        return self._layers.get(layer_index)

    def get_mask_len(self, node_name):
        nodes = self.get_indices_for_name(node_name)
        if nodes is None or len(nodes) == 0:
            return 0
        mask = self._node_edges_mask.get(nodes[0], None)
        if mask is None:
            return 0
        return len(mask)

    def get_layer_index_for_node_name(self, name):
        index = self._name_index_map.get(name)
        for key in self._layers.keys():
            layer = self._layers.get(key)
            if len(layer) == 0:
                continue
            elif layer[0] <= index <= layer[len(layer) - 1]:
                return key

    def get_indices_for_name(self, name):
        return self._name_index_map.get(name)

    def get_name_for_index(self, index):
        return self._index_name_map.get(index)

    def add_vertex(self, node_name, layer=None, **kwargs):
        if layer is None:
            layer = self.get_next_layer_index()
        next_node_index = len(self.nodes)
        self._index_name_map[next_node_index] = node_name

        if self._name_index_map.get(node_name, None) is None:
            self._name_index_map[node_name] = []
        self._name_index_map[node_name].append(next_node_index)

        mask = kwargs.pop('mask', None)
        if mask is not None:
            self._node_edges_mask[next_node_index] = mask

        if self._layers.get(layer, None) is None:
            self._layers[layer] = []
        self._layers[layer].append(next_node_index)

        super().add_node(next_node_index, name=node_name, **kwargs)

    def _flatten_args(self, nested_list):
        flat_list = []
        for element in nested_list:
            if isinstance(element, (list, tuple)):
                flat_list.extend(self._flatten_args(element))
            else:
                flat_list.append(element)
        return flat_list

    def add_edges(self, source_node_names: List, target_node_name):
        target_indices = self.get_indices_for_name(target_node_name)
        source_node_names = self._flatten_args(source_node_names)
        for source_node_name in source_node_names:
            source_indices = self._determine_source_indices(source_node_name)
            if source_indices is not None:  # ignore nodes that were not added to the graph before e.g. constants
                self._add_edges(source_indices, target_indices)

    def _determine_source_indices(self, source_node_name):
        s_name = str(source_node_name)
        if s_name in self.ignored_nodes:
            next_layer = self.get_next_layer_index() - 2 if self.get_next_layer_index() >= 2 else 0
            if next_layer > 0:
                return self._layers.get(next_layer)
            else:
                return None
        else:
            return self.get_indices_for_name(s_name)

    def _add_edges(self, source_indices, target_indices):
        for s_i in source_indices:
            source_counter = 0
            mask = self._node_edges_mask.get(s_i, None)
            if mask is None:
                for t_i in target_indices:
                    super().add_edge(s_i, t_i)
            else:
                for t_i in target_indices:
                    if mask[source_counter]:
                        super().add_edge(s_i, t_i)
                    source_counter += 1
