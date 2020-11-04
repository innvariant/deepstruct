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

    def add_edges_from(self, ebunch_to_add, **attr):
        for (u, v) in ebunch_to_add:
            self.add_edge(u, v, **attr)

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