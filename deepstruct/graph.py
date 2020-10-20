import networkx as nx


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


def build_layer_index(graph: nx.DiGraph, layer_index=None):
    """

    :param graph:
    :type graph igraph.Graph
    :param layer_index:
    :return:
    """
    if layer_index is None:
        layer_index = {}

    recursion_call = {"count": 0}

    def get_layer_index(vertex, graph):
        try:
            vertex = int(vertex)
        except TypeError:
            raise ValueError("You have to pass vertex indices to this function.")
        # print('get_layer_index(%s, roots, graph)' % vertex)
        if vertex is None:
            raise ValueError("Given vertex was none.")
        if vertex not in layer_index:
            recursion_call["count"] += 1
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

    # print('Recursion call: %s' % recursion_call['count'])

    vertices_by_layer = {}
    for v in layer_index:
        idx = layer_index[v]
        if idx not in vertices_by_layer:
            vertices_by_layer[idx] = []
        vertices_by_layer[idx].append(v)

    return layer_index, vertices_by_layer
