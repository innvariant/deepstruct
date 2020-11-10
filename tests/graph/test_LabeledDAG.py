import itertools

import numpy as np
import pytest

import deepstruct.graph


def test_add_single():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()

    # Act
    dag.add_node(5)

    # Assert
    assert len(dag.nodes) == 1
    assert 0 in dag.nodes
    assert 5 not in dag.nodes
    assert len(dag.edges) == 0
    assert dag.num_layers == 1


def test_add_two_separate_nodes():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()
    random_label_1_larger_than_one = np.random.randint(2, 5)
    random_label_2_larger_than_one = np.random.randint(2, 5)

    # Act
    dag.add_node(random_label_1_larger_than_one)
    dag.add_node(random_label_2_larger_than_one)

    # Assert
    assert len(dag.nodes) == 2
    assert 0 in dag.nodes
    assert 1 in dag.nodes
    assert random_label_1_larger_than_one not in dag.nodes
    assert random_label_2_larger_than_one not in dag.nodes
    assert len(dag.edges) == 0
    assert dag.num_layers == 1


def test_add_two_connected_nodes():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()
    random_label_1_larger_than_one = np.random.randint(2, 5)
    random_label_2_larger_than_one = np.random.randint(2, 5)

    # Act
    dag.add_edge(random_label_1_larger_than_one, random_label_2_larger_than_one)

    # Assert
    assert len(dag.nodes) == 2
    assert 0 in dag.nodes
    assert 1 in dag.nodes
    assert random_label_1_larger_than_one not in dag.nodes
    assert random_label_2_larger_than_one not in dag.nodes
    assert len(dag.edges) == 1
    assert dag.num_layers == 2


def test_cycle():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()
    dag.add_edge(0, 1)  # Add one valid edge and two nodes
    dag.add_edge(1, 2)  # Add one valid edge and one new node

    # Act
    with pytest.raises(AssertionError):
        dag.add_edge(2, 0)  # This would close the cycle

    # Assert
    assert len(dag.nodes) == 3
    assert len(dag.edges) == 2
    assert dag.num_layers == 3


def test_add_two_layers():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()

    size_layer0 = np.random.randint(2, 10)
    size_layer1 = np.random.randint(2, 10)

    layer0 = dag.add_vertices(size_layer0, layer=0)
    layer1 = dag.add_vertices(size_layer1, layer=1)

    # Act
    for source in layer0:
        dag.add_edges_from((source, t) for t in layer1)

    # Assert
    assert len(dag.nodes) == size_layer0 + size_layer1
    assert len(dag.edges) == size_layer0 * size_layer1
    assert dag.num_layers == 2


def test_add_two_layers_crossing():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()

    size_layer0 = np.random.randint(2, 10)
    size_layer1 = np.random.randint(2, 10)

    # Act
    dag.add_edges_from(
        itertools.product(
            np.arange(size_layer0), np.arange(size_layer0 + size_layer1 + 1)
        )
    )


def test_multiple_large_layers():
    # Arrange
    dag = deepstruct.graph.LabeledDAG()

    num_layers = np.random.randint(15, 21)
    size_layer = {}
    for layer in range(num_layers):
        size_layer[layer] = np.random.randint(50, 101)
        dag.add_vertices(size_layer[layer], layer=layer)

    # Act
    num_edges = 0
    for layer_source in range(num_layers - 1):
        for layer_target in np.random.choice(
            range(layer_source + 1, num_layers),
            np.random.randint(num_layers - layer_source),
            replace=False,
        ):
            v_source = np.random.choice(
                dag.get_vertices(layer_source),
                dag.get_layer_size(layer_source),
                replace=False,
            )
            v_target = np.random.choice(
                dag.get_vertices(layer_target),
                dag.get_layer_size(layer_target),
                replace=False,
            )
            dag.add_edges_from((s, t) for t in v_target for s in v_source)
            num_edges += len(v_source) * len(v_target)

    # Assert
    assert len(dag.nodes) == sum(size_layer.values())
    assert len(dag.edges) == num_edges
    assert dag.num_layers == num_layers


def test_append_simple():
    # Arrange
    graph1 = deepstruct.graph.LabeledDAG()
    graph2 = deepstruct.graph.LabeledDAG()

    graph1_size_layer0 = 3  # np.random.randint(2, 10)
    graph1_size_layer1 = 5  # np.random.randint(2, 10)
    graph2_size_layer0 = graph1_size_layer1
    graph2_size_layer1 = 4  # np.random.randint(2, 10)

    graph1_layer0 = graph1.add_vertices(graph1_size_layer0, layer=0)
    graph1_layer1 = graph1.add_vertices(graph1_size_layer1, layer=1)
    graph2_layer0 = graph2.add_vertices(graph2_size_layer0, layer=0)
    graph2_layer1 = graph2.add_vertices(graph2_size_layer1, layer=1)
    graph1.add_edges_from(
        (s, t) for t in graph1_layer1 for s in graph1_layer0 if np.random.randint(2)
    )
    graph2.add_edges_from(
        (s, t) for t in graph2_layer1 for s in graph2_layer0 if np.random.randint(3)
    )

    graph1_num_edges = len(graph1.edges)
    graph2_num_edges = len(graph2.edges)

    graph1.append(graph2)

    assert len(graph1) == graph1_size_layer0 + graph1_size_layer1 + graph2_size_layer1
    assert len(graph1.edges) == graph1_num_edges + graph2_num_edges
