import unittest
import pypaddle.util
import networkx as nx


class LayerIndexTest(unittest.TestCase):

    def test_dev(self):
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

        layer_index, vertex_by_layer = pypaddle.util.build_layer_index(structure)

        for n in nodes:
            self.assertTrue(n in layer_index)
            self.assertTrue(layer_index[n] in vertex_by_layer)
            self.assertTrue(n in vertex_by_layer[layer_index[n]])
