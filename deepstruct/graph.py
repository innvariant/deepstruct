from __future__ import annotations

import networkx as nx
import numpy as np


class LayeredGraph(nx.DiGraph):
    @property
    def first_layer(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def last_layer(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def num_layers(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def first_layer_size(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def last_layer_size(self):
        """
        :rtype: int
        """
        return NotImplementedError()

    @property
    def layers(self):
        """
        :rtype: list[int]
        """
        raise NotImplementedError()

    def get_layer(self, vertex: int):
        """
        :rtype: int
        """
        raise NotImplementedError()

    def get_vertices(self, layer: int):
        """
        :rtype: list[int]
        """
        raise NotImplementedError()

    def get_layer_size(self, layer: int):
        """
        :rtype: int
        """
        raise NotImplementedError()

    def layer_connected(self, layer_index1: int, layer_index2: int):
        """
        :rtype: bool
        """
        raise NotImplementedError()

    def layer_connection_size(self, layer_index1: int, layer_index2: int):
        """
        :rtype: int
        """
        raise NotImplementedError()


class CachedLayeredGraph(LayeredGraph):
    def __init__(self, **attr):
        super(CachedLayeredGraph, self).__init__(**attr)
        self._has_changed = True
        self._layer_index = None
        self._vertex_by_layer = None

    def add_cycle(self, nodes, **attr):
        super(LayeredGraph, self).add_cycle(nodes, **attr)
        self._has_changed = True

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        super(LayeredGraph, self).add_edge(u_of_edge, v_of_edge, **attr)
        self._has_changed = True

    def add_edges_from(self, ebunch_to_add, **attr):
        super(LayeredGraph, self).add_edges_from(ebunch_to_add, **attr)
        self._has_changed = True

    def add_node(self, node_for_adding, **attr):
        super(LayeredGraph, self).add_node(node_for_adding, **attr)
        self._has_changed = True

    def add_nodes_from(self, nodes_for_adding, **attr):
        super(LayeredGraph, self).add_nodes_from(nodes_for_adding, **attr)
        self._has_changed = True

    def add_path(self, nodes, **attr):
        super(LayeredGraph, self).add_path(nodes, **attr)
        self._has_changed = True

    def add_star(self, nodes, **attr):
        super(LayeredGraph, self).add_star(nodes, **attr)
        self._has_changed = True

    def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):
        super(LayeredGraph, self).add_weighted_edges_from(
            ebunch_to_add, weight="weight", **attr
        )
        self._has_changed = True

    def _get_layer_index(self):
        if (
            self._has_changed
            or self._layer_index is None
            or self._vertex_by_layer is None
        ):
            self._build_layer_index()
            self._has_changed = False

        return self._layer_index, self._vertex_by_layer

    def _layer_by_vertex(self, vertex: int):
        return self._get_layer_index()[0][vertex]

    def _vertices_by_layer(self, layer: int):
        return self._get_layer_index()[1][layer]

    def _build_layer_index(self):
        self._layer_index, self._vertex_by_layer = build_layer_index(self)

    @property
    def first_layer(self):
        """
        :rtype: int
        """
        return self.layers[0]

    @property
    def last_layer(self):
        """
        :rtype: int
        """
        return self.layers[-1]

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def first_layer_size(self):
        return self.get_layer_size(self.layers[0])

    @property
    def last_layer_size(self):
        return self.get_layer_size(self.layers[-1])

    @property
    def layers(self):
        return [layer for layer in self._get_layer_index()[1]]

    def get_layer(self, vertex: int):
        return self._layer_by_vertex(vertex)

    def get_vertices(self, layer: int):
        return self._vertices_by_layer(layer)

    def get_layer_size(self, layer: int):
        return len(self._vertices_by_layer(layer))

    def layer_connected(self, layer_index1: int, layer_index2: int):
        """
        :rtype: bool
        """
        if layer_index1 is layer_index2:
            raise ValueError(
                "Same layer does not have interconnections, it would be split up."
            )
        if layer_index1 > layer_index2:
            tmp = layer_index2
            layer_index2 = layer_index1
            layer_index1 = tmp

        for source_vertex in self.get_vertices(layer_index1):
            for target_vertex in self.get_vertices(layer_index2):
                if self.has_edge(source_vertex, target_vertex):
                    return True
        return False

    def layer_connection_size(self, layer_index1: int, layer_index2: int):
        """
        :rtype: int
        """
        if layer_index1 is layer_index2:
            raise ValueError(
                "Same layer does not have interconnections, it would be split up."
            )
        if layer_index1 > layer_index2:
            tmp = layer_index2
            layer_index2 = layer_index1
            layer_index1 = tmp

        size = 0
        for source_vertex in self.get_vertices(layer_index1):
            for target_vertex in self.get_vertices(layer_index2):
                if self.has_edge(source_vertex, target_vertex):
                    size += 1
        return size


class LabeledDAG(LayeredGraph):
    """
    Directed acyclic graph in which the order of vertices matters as they are enumerated.
    The implementation makes sure you add no cycles.
    """

    def __init__(self, **attr):
        super(LabeledDAG, self).__init__(**attr)
        self._layer_index = {}
        self._vertex_by_layer = {}
        self._update_indices()
        self._has_changed = False

    def _update_indices(self):
        self._layer_index, self._vertex_by_layer = build_layer_index(
            self, self._layer_index
        )

    def _get_layer_index(self):
        if (
            self._has_changed
            or self._layer_index is None
            or self._vertex_by_layer is None
        ):
            self._update_indices()
            self._has_changed = False

        return self._layer_index, self._vertex_by_layer

    def _layer_by_vertex(self, vertex: int):
        return self._get_layer_index()[0][vertex]

    def _vertices_by_layer(self, layer: int):
        return self._get_layer_index()[1][layer]

    def index_in_layer(self, vertex):
        layer_vertices = self.get_vertices(self.get_layer(vertex))
        return layer_vertices.index(vertex)
        """match = np.where(self.get_vertices(self.get_layer(vertex)) == vertex)
        return match[0][0] if len(match) > 0 else None"""

    def add_vertex(self, layer: int = 0):
        assert layer >= 0

        new_node = len(self.nodes)
        if layer not in self._vertex_by_layer:
            self._vertex_by_layer[layer] = []
        self._vertex_by_layer[layer].append(new_node)
        self._layer_index[new_node] = layer
        super().add_node(new_node)
        return new_node

    def add_vertices(self, num_vertices: int, layer: int = 0):
        assert num_vertices > 0
        assert layer >= 0

        new_nodes = np.arange(len(self), len(self) + num_vertices)
        if layer not in self._vertex_by_layer:
            self._vertex_by_layer[layer] = []
        self._vertex_by_layer[layer].extend(new_nodes)
        self._layer_index.update(dict.fromkeys(new_nodes, layer))
        super().add_nodes_from(new_nodes)
        return new_nodes

    def append(self, other: LabeledDAG):
        assert self.last_layer is not None
        assert other is not None
        assert other.first_layer is not None
        assert other.first_layer != other.last_layer
        assert self.last_layer_size == other.first_layer_size

        offset_layer = self.last_layer

        for layer in other.layers[1:]:
            own_layer_target = offset_layer + layer
            self.add_vertices(other.get_layer_size(layer), own_layer_target)
            for oth_v_idx, oth_v in enumerate(other.get_vertices(layer)):
                own_v = self.get_vertices(own_layer_target)[oth_v_idx]
                for (oth_u, _) in other.in_edges(oth_v):
                    oth_u_idx = other.index_in_layer(oth_u)
                    own_layer_source = offset_layer + other.get_layer(oth_u)
                    own_u = self.get_vertices(own_layer_source)[oth_u_idx]
                    self.add_edge(own_u, own_v)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        new_layer_source = (
            0 if "source_layer" not in attr else int(attr["source_layer"])
        )
        source = (
            u_of_edge
            if u_of_edge in self.nodes
            else self.add_vertex(layer=new_layer_source)
        )
        layer_source = self.get_layer(source)
        new_layer_target = (
            max(1, layer_source + 1)
            if "target_layer" not in attr
            else int(attr["target_layer"])
        )
        target = (
            v_of_edge
            if v_of_edge in self.nodes
            else self.add_vertex(layer=new_layer_target)
        )
        layer_target = self.get_layer(target)
        assert (
            layer_source < layer_target
        ), "Can only add edges from lower layers numbers to higher layer numbers. We found L({source})={slayer} >= L({target})={tlayer}".format(
            source=source, slayer=layer_source, target=target, tlayer=layer_target
        )
        super().add_edge(source, target, **attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        return self.add_vertices(len(nodes_for_adding))

    def add_node(self, node_for_adding, **attr):
        return self.add_vertex(layer=0 if "layer" not in attr else attr["layer"])

    def _get_next_layer_or_param(self, layer_source: int, **attr):
        return (
            max(1, layer_source + 1)
            if "target_layer" not in attr
            else int(attr["target_layer"])
        )

    def add_edges_from(self, ebunch_to_add, **attr):
        new_layer_source = (
            0 if "source_layer" not in attr else int(attr["source_layer"])
        )
        source_map = {}
        target_map = {}
        edges = []
        for (s, t) in ebunch_to_add:
            if s not in source_map:
                source_map[s] = (
                    s if s in self.nodes else self.add_vertex(new_layer_source)
                )
            if t not in target_map:
                target_map[t] = (
                    t
                    if t in self.nodes
                    else self.add_vertex(
                        self._get_next_layer_or_param(
                            self.get_layer(source_map[s]), **attr
                        )
                    )
                )
            edges.append((source_map[s], target_map[t]))
        super().add_edges_from(edges)

    @property
    def first_layer(self):
        """
        :rtype: int
        """
        return self.layers[0] if self.num_layers > 0 else None

    @property
    def last_layer(self):
        """
        :rtype: int
        """
        return self.layers[-1] if self.num_layers > 0 else None

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def first_layer_size(self):
        return self.get_layer_size(self.layers[0])

    @property
    def last_layer_size(self):
        return self.get_layer_size(self.layers[-1])

    @property
    def layers(self):
        return [layer for layer in self._get_layer_index()[1]]

    def get_layer(self, vertex: int):
        return self._layer_by_vertex(vertex)

    def get_vertices(self, layer: int):
        return self._vertices_by_layer(layer)

    def get_layer_size(self, layer: int):
        return len(self._vertices_by_layer(layer))

    def layer_connected(self, layer_index1: int, layer_index2: int):
        """
        :rtype: bool
        """
        if layer_index1 is layer_index2:
            raise ValueError(
                "Same layer does not have interconnections, it would be split up."
            )
        if layer_index1 > layer_index2:
            tmp = layer_index2
            layer_index2 = layer_index1
            layer_index1 = tmp

        for source_vertex in self.get_vertices(layer_index1):
            for target_vertex in self.get_vertices(layer_index2):
                if self.has_edge(source_vertex, target_vertex):
                    return True
        return False

    def layer_connection_size(self, layer_index1: int, layer_index2: int):
        """
        :rtype: int
        """
        if layer_index1 is layer_index2:
            raise ValueError(
                "Same layer does not have interconnections, it would be split up."
            )
        if layer_index1 > layer_index2:
            tmp = layer_index2
            layer_index2 = layer_index1
            layer_index1 = tmp

        size = 0
        for source_vertex in self.get_vertices(layer_index1):
            for target_vertex in self.get_vertices(layer_index2):
                if self.has_edge(source_vertex, target_vertex):
                    size += 1
        return size


class MarkableDAG(LabeledDAG):
    def add_connection(self, mark: str, s_idx: int, t_idx: int):
        pass


def build_layer_index(graph: nx.DiGraph, layer_index=None):
    """

    :param graph:
    :type graph igraph.Graph
    :param layer_index:
    :return:
    """
    if layer_index is None:
        layer_index = {}

    def get_layer_index(vertex, graph: nx.DiGraph):
        assert vertex is not None, "Given vertex was none."
        try:
            vertex = int(vertex)
        except TypeError:
            raise ValueError("You have to pass vertex indices to this function.")

        if vertex not in layer_index:
            # Recursively calling itself
            layer_index[vertex] = (
                max(
                    [
                        get_layer_index(v, graph)
                        for v in nx.algorithms.dag.ancestors(graph, vertex)
                    ]
                    + [-1]
                )
                + 1
            )
        return layer_index[vertex]

    for v in graph:
        get_layer_index(v, graph)

    vertices_by_layer = {}
    for v in layer_index:
        idx = layer_index[v]
        if idx not in vertices_by_layer:
            vertices_by_layer[idx] = []
        vertices_by_layer[idx].append(v)

    return layer_index, vertices_by_layer
