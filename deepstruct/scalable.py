import itertools

import numpy as np
import torch
import torch.nn as nn

from deepstruct.graph import CachedLayeredGraph
from deepstruct.graph import LayeredGraph
from deepstruct.sparse import MaskedDeepDAN
from deepstruct.sparse import MaskedDeepFFN
from deepstruct.util import kullback_leibler


class ScalableDeepFFN(object):
    _precision: int = 2
    _epsilon: float = 0.001
    _entropy_equivalence_epsilon: float = 0.1

    def __init__(self, proportions: np.ndarray):
        assert proportions is not None
        assert len(proportions) > 0
        assert np.abs(np.sum(proportions) - 1) < self._epsilon

        self._proportions = proportions

    @property
    def entropy_similarity_epsilon(self) -> float:
        return self._entropy_equivalence_epsilon

    @entropy_similarity_epsilon.setter
    def entropy_similarity_epsilon(self, epsilon):
        assert epsilon > 0
        self._entropy_equivalence_epsilon = epsilon

    @property
    def precision(self) -> int:
        return self._precision

    @precision.setter
    def precision(self, precision: int):
        assert precision > 0
        self._precision = precision

    @property
    def proportions(self):
        return np.round(self._proportions, self.precision)

    def draw(self, scale: int) -> list:
        return [round(int(np.maximum(1, size))) for size in scale * self.proportions]

    def build(self, input_shape, output_shape, scale: int) -> nn.Module:
        assert scale > 0

        layers = self.draw(scale)
        print("l", layers, np.sum(layers))
        return MaskedDeepFFN(input_shape, output_shape, layers)

    def __eq__(self, other):
        if not isinstance(other, ScalableDeepFFN):
            return False

        return self.entropy_similarity(other) < self.entropy_similarity_epsilon

    def entropy_similarity(self, other) -> float:
        assert isinstance(other, ScalableDeepFFN)
        max_length = np.maximum(len(self._proportions), len(other._proportions))
        return kullback_leibler(
            np.pad(self._proportions, (0, max_length - len(self._proportions))),
            np.pad(other._proportions, (0, max_length - len(other._proportions))),
        )

    def __str__(self):
        return str(self.draw(np.power(10, self.precision)))


class ScalableDAN(object):
    _cached_scaled_structure: LayeredGraph = None
    _vertex_correspondences: dict = None

    _structure: LayeredGraph
    _proportion_map: dict

    def __init__(self, structure: LayeredGraph, proportion_map: dict):
        if proportion_map is None:
            proportion_map = {v: 1 / len(structure.nodes) for v in structure.nodes}
        self.structure = structure
        self.proportions = proportion_map

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        assert structure is not None
        assert len(structure.nodes) > 0

        self._structure = structure

    @property
    def proportions(self):
        return self._proportion_map

    @proportions.setter
    def proportions(self, map):
        assert len(map) == len(self.structure.nodes)
        assert np.isclose(sum(map.values()), 1)
        self._proportion_map = map

    def grow(self, scale: int) -> LayeredGraph:
        assert scale > 0

        for layer in self.structure.layers:
            for v in self.structure.get_vertices(layer):
                self.scale(v, int(np.round(self.proportions[v] * scale)))

        return self._cached_scaled_structure

    def scale(self, vertex, size: int):
        graph_scaled = (
            self._cached_scaled_structure
            if self._cached_scaled_structure is not None
            else CachedLayeredGraph()
        )
        vertex_correspondences = (
            self._vertex_correspondences
            if self._vertex_correspondences is not None
            else {}
        )
        nodes_offset = len(graph_scaled.nodes)

        if vertex not in vertex_correspondences:
            vertex_correspondences[vertex] = np.array([])

        vertex_correspondences[vertex] = np.concatenate(
            [
                vertex_correspondences[vertex],
                np.arange(nodes_offset, nodes_offset + size),
            ]
        )
        graph_scaled.add_nodes_from(
            vertex_correspondences[vertex]
        )  # ['v%s_%s' % (v, idx) for idx in range(size)])
        nodes_offset += size

        for (source_vertex, _) in self.structure.in_edges(vertex):
            graph_scaled.add_edges_from(
                itertools.product(
                    vertex_correspondences[source_vertex],
                    vertex_correspondences[vertex],
                )
            )

        self._cached_scaled_structure = graph_scaled
        self._vertex_correspondences = vertex_correspondences

        return graph_scaled

    def build(
        self, input_shape, output_shape, scale: int, use_layer_norm: bool = True
    ) -> nn.Module:
        assert scale > 0
        graph_scaled = self.grow(scale)
        self._cached_scaled_structure = None
        self._vertex_correspondences = None
        return MaskedDeepDAN(
            input_shape, output_shape, graph_scaled, use_layer_norm=use_layer_norm
        )


if __name__ == "__main__":
    g1 = CachedLayeredGraph()
    g1.add_edge(0, 3)
    g1.add_edge(0, 5)
    g1.add_edge(1, 3)
    g1.add_edge(1, 4)
    g1.add_edge(1, 5)
    g1.add_edge(2, 3)
    g1.add_edge(2, 4)

    g1.add_edge(3, 5)
    g1.add_edge(3, 6)
    g1.add_edge(4, 5)
    g1.add_edge(4, 7)

    g1.add_edge(5, 7)

    props = {
        v: p
        for v, p in zip(g1.nodes, np.random.dirichlet(np.ones(len(g1.nodes)) * 100))
    }
    print(props)
    fam = ScalableDAN(g1, props)
    model = fam.build(8, 4, 1000)
    print(model)
    for lay in model.layers_main_hidden:
        print(lay)
        print(torch.sum(lay.mask) / np.prod(lay.mask.shape))

    for lay in model.layers_skip_hidden:
        print(lay)
        print(torch.sum(lay.mask) / np.prod(lay.mask.shape))
