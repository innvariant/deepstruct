import unittest
import numpy as np
import pypaddle.util
import pypaddle.sparse
import networkx as nx

class MaskedDeepDANTest(unittest.TestCase):
    def test_traversal(self):
        # Arrange
        random_graph = nx.watts_strogatz_graph(200, 3, 0.8)
        structure = pypaddle.sparse.CachedLayeredGraph()
        structure.add_edges_from(random_graph.edges)
        structure.add_nodes_from(random_graph.nodes)

        print([structure.in_degree(n) for n in structure.nodes])

    def test_random_structures_success(self):
        # Arrange
        random_graph = nx.watts_strogatz_graph(200, 3, 0.8)
        structure = pypaddle.sparse.CachedLayeredGraph()
        structure.add_edges_from(random_graph.edges)
        structure.add_nodes_from(random_graph.nodes)
        model = pypaddle.sparse.MaskedDeepDAN(784, 10, structure)

        # Act
        extracted_structure = model.generate_structure()
        new_model = pypaddle.sparse.MaskedDeepDAN(784, 10, extracted_structure)

        # Assert
        #self.assertTrue(nx.algorithms.isomorphism.faster_could_be_isomorphic(structure, extracted_structure))
        self.assertTrue(nx.is_isomorphic(structure, extracted_structure))
        self.assertTrue(nx.is_isomorphic(structure, new_model.generate_structure()))

    def test_random_structures_with_input_and_output_success(self):
        # Arrange
        random_graph = nx.watts_strogatz_graph(200, 3, 0.8)
        structure = pypaddle.sparse.CachedLayeredGraph()
        structure.add_edges_from(random_graph.edges)
        structure.add_nodes_from(random_graph.nodes)
        model = pypaddle.sparse.MaskedDeepDAN(784, 10, structure)

        # Act
        extracted_structure = model.generate_structure(include_input=True, include_output=True)
        new_model = pypaddle.sparse.MaskedDeepDAN(784, 10, extracted_structure)

    def test_apply_mask_success(self):
        random_graph = nx.watts_strogatz_graph(200, 3, 0.8)
        structure = pypaddle.sparse.CachedLayeredGraph()
        structure.add_edges_from(random_graph.edges)
        structure.add_nodes_from(random_graph.nodes)
        model = pypaddle.sparse.MaskedDeepDAN(784, 10, structure)

        previous_weights = []
        for layer in pypaddle.sparse.maskable_layers(model):
            previous_weights.append(np.copy(layer.weight.detach().numpy()))

        model.apply_mask()

        different = []
        for layer, previous_weight in zip(pypaddle.sparse.maskable_layers(model), previous_weights):
            different.append(not np.all(np.equal(np.array(previous_weight), np.array(layer.weight.detach().numpy()))))
        self.assertTrue(np.any(different))


    def test_get_structure(self):
        structure = pypaddle.sparse.CachedLayeredGraph()

        block0_size = 8
        block1_size = 8
        block2_size = 2
        block3_size = 2
        block4_size = 2
        block5_size = 2
        block6_size = 10
        block0 = np.arange(1, block0_size+1)
        block1 = np.arange(block0_size+1, block0_size+block1_size+1)
        block2 = np.arange(block0_size+block1_size+1, block0_size+block1_size+block2_size+1)
        block3 = np.arange(block0_size+block1_size+block2_size+1, block0_size+block1_size+block2_size+block3_size+1)
        block4 = np.arange(block0_size+block1_size+block2_size+block3_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+1)
        block5 = np.arange(block0_size+block1_size+block2_size+block3_size+block4_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+1)
        block6 = np.arange(block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+block6_size+1)

        # First layer
        for v in block0:
            for t in block2:
                structure.add_edge(v, t)
        for v in block0:
            for t in block3:
                structure.add_edge(v, t)
        for v in block0:
            for t in block5:
                structure.add_edge(v, t)
        for v in block1:
            for t in block3:
                structure.add_edge(v, t)
        for v in block1:
            for t in block4:
                structure.add_edge(v, t)
        for v in block1:
            for t in block6:
                structure.add_edge(v, t)

        # Second layer
        for v in block2:
            for t in block5:
                structure.add_edge(v, t)
        for v in block3:
            for t in block5:
                structure.add_edge(v, t)
        for v in block3:
            for t in block6:
                structure.add_edge(v, t)
        for v in block4:
            for t in block6:
                structure.add_edge(v, t)

        model = pypaddle.sparse.MaskedDeepDAN(784, 10, structure)
        print(model)

        new_structure = model.generate_structure(include_input=False, include_output=False)

        model2 = pypaddle.sparse.MaskedDeepDAN(784, 10, new_structure)
        print(model2)

    def test_dev(self):
        structure = pypaddle.sparse.CachedLayeredGraph()
        structure.add_nodes_from(np.arange(1, 7))

        block0_size = 50
        block1_size = 50
        block2_size = 30
        block3_size = 30
        block4_size = 30
        block5_size = 20
        block6_size = 20
        block0 = np.arange(1, block0_size+1)
        block1 = np.arange(block0_size+1, block0_size+block1_size+1)
        block2 = np.arange(block0_size+block1_size+1, block0_size+block1_size+block2_size+1)
        block3 = np.arange(block0_size+block1_size+block2_size+1, block0_size+block1_size+block2_size+block3_size+1)
        block4 = np.arange(block0_size+block1_size+block2_size+block3_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+1)
        block5 = np.arange(block0_size+block1_size+block2_size+block3_size+block4_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+1)
        block6 = np.arange(block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+1, block0_size+block1_size+block2_size+block3_size+block4_size+block5_size+block6_size+1)

        # First layer
        for v in block0:
            for t in block2:
                structure.add_edge(v, t)
        for v in block0:
            for t in block3:
                structure.add_edge(v, t)
        for v in block1:
            for t in block3:
                structure.add_edge(v, t)
        for v in block1:
            for t in block4:
                structure.add_edge(v, t)

        # Second layer
        for v in block2:
            for t in block5:
                structure.add_edge(v, t)
        for v in block3:
            for t in block5:
                structure.add_edge(v, t)
        for v in block3:
            for t in block6:
                structure.add_edge(v, t)
        for v in block4:
            for t in block6:
                structure.add_edge(v, t)

        model = pypaddle.sparse.MaskedDeepDAN(784, 10, structure)