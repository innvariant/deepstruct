import networkx as nx

import deepstruct.graph
import deepstruct.util


def test_dev():
    nodes = [1, 2, 3, 4, 5]

    structure = nx.DiGraph()
    structure.add_nodes_from(nodes)
    structure.add_edge(1, 3)
    structure.add_edge(1, 4)
    structure.add_edge(1, 5)
    structure.add_edge(2, 3)
    structure.add_edge(2, 4)
    structure.add_edge(3, 5)
    structure.add_edge(4, 5)

    layer_index, vertex_by_layer = deepstruct.graph.build_layer_index(structure)

    for n in nodes:
        assert n in layer_index
        assert layer_index[n] in vertex_by_layer
        assert n in vertex_by_layer[layer_index[n]]
