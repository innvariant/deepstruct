from typing import List

import networkx


class LayeredFXGraph(networkx.DiGraph):

    def __init__(self, **attr):
        super(LayeredFXGraph, self).__init__(**attr)
        self._node_name_data = {}  # information for a name -> [layer_number, [indices], output_layer_size]
        self._mask_for_name = {}
        self.edges_for_name = {}
        self.ignored_nodes = []

    def get_next_layer_index(self):
        return len(self._node_name_data)

    def get_output_layer_len(self, node_name):
        data = self._node_name_data.get(node_name, None)
        return data[2] if data is not None else 0

    def get_layer_number(self, node_name):
        data = self._node_name_data.get(node_name, None)
        return data[0] if data is not None else None

    def get_indices_for_name(self, node_name):
        data = self._node_name_data.get(node_name, None)
        return data[1] if data is not None else None

    def add_vertices(self, count: int, name, output_layer_size=0, layer=None, **kwargs):
        node_data = []
        node_indices = []
        mask = kwargs.pop('mask', None)
        if mask is not None:
            self._mask_for_name[name] = mask
        if layer is None:
            layer = self.get_next_layer_index()
        node_data.append(layer)
        for _ in range(count):
            node_indices.append(self._add_vertex(name, **kwargs))
        node_data.append(node_indices)
        node_data.append(output_layer_size)
        self._node_name_data[name] = node_data
        return node_indices

    def _add_vertex(self, node_name, **kwargs):
        next_node_index = len(self.nodes)
        super().add_node(next_node_index, name=node_name, **kwargs)
        return next_node_index

    def add_edges(self, source_node_names: List, target_node_name):
        target_indices = self.get_indices_for_name(target_node_name)
        source_node_names = self._flatten_args(source_node_names)
        for source_node_name in source_node_names:
            s_n = str(source_node_name)
            edges = self.edges_for_name.pop(s_n, None)
            source_indices = self._determine_source_indices(s_n)
            if source_indices is not None:  # ignore nodes that were not added to the graph before e.g. constants
                self._add_edges(source_indices, target_indices, self._mask_for_name.pop(s_n, None), edges)

    def _flatten_args(self, nested_list):
        flat_list = []
        for element in nested_list:
            if isinstance(element, (list, tuple)):
                flat_list.extend(self._flatten_args(element))
            else:
                flat_list.append(element)
        return flat_list

    def _determine_source_indices(self, source_node_name):
        if source_node_name in self.ignored_nodes:
            values = list(self._node_name_data.values())
            assert len(values) > 1
            penultimate = values[-2]
            return penultimate[1]  # the layer that was added before the current node
        else:
            return self.get_indices_for_name(source_node_name)

    def _add_edges(self, source_indices, target_indices, mask, edges):
        if edges is not None:
            super().add_edges_from(edges)
        elif mask is None:
            for s_i in source_indices:
                for t_i in target_indices:
                    super().add_edge(s_i, t_i)
        else:
            target_counter = 0
            for t_i in target_indices:
                mask_slice = mask[target_counter]
                target_counter += 1
                source_counter = 0
                for s_i in source_indices:
                    if mask_slice.numel() > 0 and mask_slice[source_counter]:
                        super().add_edge(s_i, t_i)
                    source_counter += 1
