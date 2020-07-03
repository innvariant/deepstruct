import numpy as np

import pypaddle.sparse


def test_cached_layered_graph_default_success():
    layered_graph = pypaddle.sparse.CachedLayeredGraph()

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
