import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import pypaddle.util
from torch.autograd import Variable


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

    def add_weighted_edges_from(self, ebunch_to_add, weight='weight', **attr):
        super(LayeredGraph, self).add_weighted_edges_from(ebunch_to_add, weight='weight', **attr)
        self._has_changed = True

    def _get_layer_index(self):
        if self._has_changed or self._layer_index is None or self._vertex_by_layer is None:
            self._build_layer_index()
            self._has_changed = False

        return self._layer_index, self._vertex_by_layer

    def _layer_by_vertex(self, vertex: int):
        return self._get_layer_index()[0][vertex]

    def _vertices_by_layer(self, layer: int):
        return self._get_layer_index()[1][layer]

    def _build_layer_index(self):
        self._layer_index, self._vertex_by_layer = pypaddle.util.build_layer_index(self)

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
            raise ValueError('Same layer does not have interconnections, it would be split up.')
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
            raise ValueError('Same layer does not have interconnections, it would be split up.')
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


class DeepCellDAN(nn.Module):
    def __init__(self, num_classes, input_channel_size, cell_constructor, structure: LayeredGraph):
        super(DeepCellDAN, self).__init__()

        self._structure = structure
        assert structure.num_layers > 0

        self._input_nodes = [n for n in structure.nodes if structure.in_degree(n) == 0]
        self._output_nodes = [n for n in structure.nodes if structure.out_degree(n) == 0]
        assert len(self._input_nodes) > 0, 'No input nodes in structure: len=%d' % len(self.input_nodes)
        assert len(self._output_nodes) > 0, 'No output nodes in structure: len=%d' % len(self.output_nodes)

        self._node_cells = {
            node: cell_constructor(
                is_input=node in self._input_nodes,
                is_output=node in self._output_nodes,
                in_degree=structure.in_degree(node),
                out_degree=structure.out_degree(node),
                layer=structure.get_layer(node)
            )
            for node in self._structure.nodes
        }
        self._nodes = nn.ModuleList(list(self._node_cells.values()))


class MaskableModule(nn.Module):
    def apply_mask(self):
        for layer in maskable_layers(self):
            layer.apply_mask()

    def recompute_mask(self, theta=0.0001):
        for layer in maskable_layers(self):
            layer.recompute_mask(theta)


class MaskedDeepDAN(MaskableModule):
    """
    A deep directed acyclic network model which is capable of masked layers and masked skip-layer connections.
    """
    def __init__(self, input_size, num_classes, structure : LayeredGraph):
        super(MaskedDeepDAN, self).__init__()

        self._structure = structure
        assert structure.num_layers > 0

        # Multiple dimensions for input size are flattened out
        if type(input_size) is tuple or type(input_size) is torch.Size:
            input_size = np.prod(input_size)
        input_size = int(input_size)

        self.layer_first = MaskedLinearLayer(input_size, structure.first_layer_size)
        self.layers_main_hidden = nn.ModuleList([MaskedLinearLayer(structure.get_layer_size(l-1), structure.get_layer_size(l)) for l in structure.layers[1:]])

        for layer_idx, layer in zip(structure.layers[1:], self.layers_main_hidden):
            mask = torch.zeros(structure.get_layer_size(layer_idx), structure.get_layer_size(layer_idx-1))
            for source_idx, source_vertex in enumerate(structure.get_vertices(layer_idx-1)):
                for target_idx, target_vertex in enumerate(structure.get_vertices(layer_idx)):
                    if structure.has_edge(source_vertex, target_vertex):
                        mask[target_idx][source_idx] = 1
            layer.set_mask(mask)

        skip_layers = []
        self._skip_targets = {}
        for target_layer in structure.layers[2:]:
            target_size = structure.get_layer_size(target_layer)
            for distant_source_layer in structure.layers[:target_layer-1]:
                if structure.layer_connected(distant_source_layer, target_layer):
                    if target_layer not in self._skip_targets:
                        self._skip_targets[target_layer] = []

                    skip_layer = MaskedLinearLayer(structure.get_layer_size(distant_source_layer), target_size)
                    mask = torch.zeros(structure.get_layer_size(target_layer), structure.get_layer_size(distant_source_layer))
                    for source_idx, source_vertex in enumerate(structure.get_vertices(distant_source_layer)):
                        for target_idx, target_vertex in enumerate(structure.get_vertices(target_layer)):
                            if structure.has_edge(source_vertex, target_vertex):
                                mask[target_idx][source_idx] = 1
                    skip_layer.set_mask(mask)

                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append({'layer': skip_layer, 'source': distant_source_layer})
        self.layers_skip_hidden = nn.ModuleList(skip_layers)

        self.layer_out = MaskedLinearLayer(structure.last_layer_size, num_classes)
        self.activation = nn.ReLU()

    def generate_structure(self, include_input=False, include_output=False):
        structure = CachedLayeredGraph()

        node_number_offset = 0
        layer_nodeidx2node = {}

        if include_input:
            layer = self.layer_first
            layer_nodeidx2node[0] = {node_idx: node for node_idx, node in enumerate(np.arange(node_number_offset, node_number_offset + layer.mask.shape[1]))}
            node_number_offset += layer.mask.shape[1]
            structure.add_nodes_from(layer_nodeidx2node[0].values())

            for source_node_idx in range(layer.mask.shape[1]):
                source_node = layer_nodeidx2node[0][source_node_idx]

                layer_nodeidx2node[1] = {node_idx: node for node_idx, node in enumerate(np.arange(node_number_offset, node_number_offset + layer.mask.shape[0]))}
                node_number_offset += layer.mask.shape[0]
                structure.add_nodes_from(layer_nodeidx2node[1].values())

                for target_node_idx in range(layer.mask.shape[0]):
                    if layer.mask[target_node_idx][source_node_idx]:
                        target_node = layer_nodeidx2node[1][target_node_idx]
                        structure.add_edge(source_node, target_node)

        # Main hidden layers
        for source_layer_idx, layer in enumerate(self.layers_main_hidden, start=1):
            if source_layer_idx not in layer_nodeidx2node:
                layer_nodeidx2node[source_layer_idx] = {node_idx: node for node_idx, node in enumerate(np.arange(node_number_offset, node_number_offset + layer.mask.shape[1]))}
                node_number_offset += layer.mask.shape[1]
                structure.add_nodes_from(layer_nodeidx2node[source_layer_idx].values())

            target_layer_idx = source_layer_idx+1
            if target_layer_idx not in layer_nodeidx2node:
                layer_nodeidx2node[target_layer_idx] = {node_idx: node for node_idx, node in enumerate(np.arange(node_number_offset, node_number_offset + layer.mask.shape[0]))}
                node_number_offset += layer.mask.shape[0]
                structure.add_nodes_from(layer_nodeidx2node[target_layer_idx].values())

            for source_node_idx in range(layer.mask.shape[1]):
                source_node = layer_nodeidx2node[source_layer_idx][source_node_idx]

                for target_node_idx in range(layer.mask.shape[0]):
                    target_node = layer_nodeidx2node[target_layer_idx][target_node_idx]

                    if layer.mask[target_node_idx][source_node_idx]:
                        structure.add_edge(source_node, target_node)

        # Skip layers
        for target_idx in self._skip_targets:
            for target in self._skip_targets[target_idx]:
                source_layer = target['layer']
                # We have to shift target layer index as layer_nodeidx2node is indexed differently
                target_layer_idx = target_idx + 1
                # Also shift the source layer index by one as we indexed layer_nodeidx2node differently
                source_layer_idx = target['source'] + 1

                if source_layer_idx not in layer_nodeidx2node:
                    # Possibly the source layer (e.g. input layer) was left out, so skip connections from it can not be
                    # considered
                    continue

                source_idx2node = layer_nodeidx2node[source_layer_idx]
                target_idx2node = layer_nodeidx2node[target_layer_idx]

                for source_node_idx in range(source_layer.mask.shape[1]):
                    source_node = source_idx2node[source_node_idx]
                    for target_node_idx in range(source_layer.mask.shape[0]):
                        if source_layer.mask[target_node_idx][source_node_idx]:
                            target_node = target_idx2node[target_node_idx]
                            structure.add_edge(source_node, target_node)

        if include_output:
            layer = self.layer_out
            layer_indices = len(layer_nodeidx2node)
            layer_nodeidx2node[layer_indices] = {node_idx: node for node_idx, node in enumerate(np.arange(node_number_offset, node_number_offset + layer.mask.shape[0]))}
            node_number_offset += layer.mask.shape[0]
            structure.add_nodes_from(layer_nodeidx2node[layer_indices].values())

            for source_node_idx in range(layer.mask.shape[1]):
                source_node = layer_nodeidx2node[layer_indices-1][source_node_idx]
                for target_node_idx in range(layer.mask.shape[0]):
                    if layer.mask[target_node_idx][source_node_idx]:
                        target_node = layer_nodeidx2node[layer_indices][target_node_idx]
                        structure.add_edge(source_node, target_node)

        return structure

    def forward(self, input):
        # input : [batch_size, ?, ?, ..], e.g. [100, 1, 28, 28] or [100, 3, 32, 32]
        x = input.flatten(start_dim=1)
        # x: [batch_size, ?*?*..], e.g. [100, 784] or [100, 3072]
        last_output = self.activation(self.layer_first(x))
        layer_results = {0: last_output}
        for layer, layer_idx in zip(self.layers_main_hidden, self._structure.layers[1:]):
            out = self.activation(layer(last_output))

            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target['layer']
                    source_idx = skip_target['source']

                    out += self.activation(source_layer(layer_results[source_idx]))

            layer_results[layer_idx] = out  # copy?
            last_output = out

        return self.layer_out(last_output)


class MaskedDeepFFN(MaskableModule):
    """
    A deep feed-forward network model which is capable of masked layers.
    Masked layers can represent sparse structures between consecutive layers.
    This representation is suitable for feed-forward sparse networks, probably with density 0.5 and above per layer.
    """
    def __init__(self, input_size, num_classes, hidden_layers : list):
        super(MaskedDeepFFN, self).__init__()
        assert len(hidden_layers) > 0

        # Multiple dimensions for input size are flattened out
        if type(input_size) is tuple or type(input_size) is torch.Size:
            input_size = np.prod(input_size)
        input_size = int(input_size)

        self.layer_first = MaskedLinearLayer(input_size, hidden_layers[0])
        self.layers_hidden = nn.ModuleList([MaskedLinearLayer(hidden_layers[l], h) for l, h in enumerate(hidden_layers[1:])])
        self.layer_out = MaskedLinearLayer(hidden_layers[-1], num_classes)
        self.activation = nn.ReLU()

    def generate_structure(self, include_input=False, include_output=False):
        """

        :param include_input:
        :param include_output:
        :rtype : LayeredGraph
        :return:
        """
        structure = CachedLayeredGraph()

        def add_edges(structure, layer, offset_source):
            offset_target = offset_source + layer.mask.shape[1]
            for source_node_idx, source_node in enumerate(range(layer.mask.shape[1])):
                for target_node_idx, target_node in enumerate(range(layer.mask.shape[0])):
                    if layer.mask[target_node_idx][source_node_idx]:
                        structure.add_edge(offset_source + source_node, offset_target + target_node)
            return offset_target

        offset_source = 0
        if include_input:
            offset_source = add_edges(structure, self.layer_first, 0)

        for layer in self.layers_hidden:
            offset_source = add_edges(structure, layer, offset_source)

        if include_output:
            add_edges(structure, self.layer_out, offset_source)

        return structure

    def forward(self, input):
        # input : [batch_size, ?, ?, ..], e.g. [100, 1, 28, 28] or [100, 3, 32, 32]
        out = self.activation(self.layer_first(input.flatten(start_dim=1)))  # [B, n_hidden_1]
        for layer in self.layers_hidden:
            out = self.activation(layer(out))
        return self.layer_out(out)  # [B, n_out]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.layer_first.to(*args, **kwargs)
        for idx, h in enumerate(self.layers_hidden):
            self.layers_hidden[idx] = self.layers_hidden[idx].to(*args, **kwargs)
        self.layer_out.to(*args, **kwargs)
        return self


def maskable_layers(network):
    for child in network.children():
        if type(child) is MaskedLinearLayer:
            yield child
        elif type(child) is nn.ModuleList:
            for layer in maskable_layers(child):
                yield layer


def maskable_layers_with_name(network):
    for name, child in network.named_children():
        if type(child) is MaskedLinearLayer:
            yield name, child
        elif type(child) is nn.ModuleList:
            for name, layer in maskable_layers_with_name(child):
                yield name, layer


def prunable_layers(network):
    return maskable_layers(network)


def prunable_layers_with_name(network):
    return maskable_layers_with_name(network)


class MaskedLinearLayer(nn.Linear):
    def __init__(self, in_feature, out_features, bias=True, keep_layer_input=False):
        """
        :param in_feature:          The number of features that are inserted in the layer.
        :param out_features:        The number of features that are returned by the layer.
        :param bias:                Iff each neuron in the layer should have a bias unit as well.
        """
        super().__init__(in_feature, out_features, bias)

        self.register_buffer('mask', torch.ones((out_features, in_feature), dtype=torch.bool))
        self.keep_layer_input = keep_layer_input
        self.layer_input = None

    def get_mask(self):
        return self.mask

    def set_mask(self, mask):
        self.mask = Variable(mask)

    def get_weight_count(self):
        return self.mask.sum()

    def apply_mask(self):
        # Assigning "self.weight = torch.nn.Parameter(self.weight.mul(self.mask))" might have side effects
        # Using direct manipulation on tensor "self.weight.data"
        self.weight.data = self.weight * self.mask

    def recompute_mask(self, theta: float = 0.001):
        """
        Recomputes the mask based on the weight magnitudes.
        If you want to consider the existing mask, make sure to first call apply_mask() and then recompute it.

        :param theta: Specifies a possible threshold for absolute distance to zero.
        """
        self.mask = torch.ones(self.weight.shape, dtype=torch.bool, device=self.mask.device)
        self.mask[torch.where(abs(self.weight) < theta)] = False

    def get_weight(self):
        return self.weight

    def reset_parameters(self, keep_mask=False):
        super().reset_parameters()
        # hasattr() is necessary because reset_parameters() is called in __init__ of Linear(), but buffer 'mask'
        # may only be registered after super() call, thus 'mask' might not be defined as buffer / attribute, yet
        if hasattr(self, 'mask') and not keep_mask:
            self.mask = torch.ones(self.weight.size(), dtype=torch.bool)

    def forward(self, input):
        x = input.float()  # In case we get a double-tensor passed, force it to be float for multiplications to work

        # Possibly store the layer input
        if self.keep_layer_input:
            self.layer_input = x.data

        return F.linear(x, self.weight * self.mask, self.bias)


