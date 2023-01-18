import os

import networkx as nx
import numpy as np

import deepstruct.graph


def test_store_load(tmp_path):
    path_graph = os.path.join(tmp_path, "tmp.graphml")

    l0 = deepstruct.graph.CachedLayeredGraph()
    l0.add_nodes_from([1, 2, 3, 4, 5])
    l0.add_edges_from([(1, 3), (2, 4), (3, 4), (4, 5)])
    l0.save(path_graph)

    l1 = deepstruct.graph.LayeredGraph.load(path_graph)

    assert l1 is not None
    assert len(l1.nodes) == len(l0.nodes)
    assert len(l1.edges) == len(l0.edges)
    assert nx.is_isomorphic(l0, l1)


def test_cached_layered_graph_default_success():
    layered_graph = deepstruct.graph.CachedLayeredGraph()

    layered_graph.add_nodes_from(np.arange(1, 7))

    # First layer
    layered_graph.add_edge(1, 3)
    layered_graph.add_edge(1, 4)
    layered_graph.add_edge(1, 5)
    layered_graph.add_edge(1, 6)
    layered_graph.add_edge(2, 3)
    layered_graph.add_edge(2, 4)
    layered_graph.add_edge(2, 5)
    layered_graph.add_edge(2, 7)

    # Second layer
    layered_graph.add_edge(3, 6)
    layered_graph.add_edge(4, 6)
    layered_graph.add_edge(4, 7)
    layered_graph.add_edge(5, 7)

    first_layer_size_before = layered_graph.get_layer_size(0)
    assert not layered_graph._has_changed

    # Add vertex 0 and connect it to vertices from layer 2
    layered_graph.add_edge(0, 3)
    layered_graph.add_edge(0, 4)
    assert layered_graph._has_changed

    first_layer_size_after = layered_graph.get_layer_size(0)
    assert first_layer_size_before != first_layer_size_after
