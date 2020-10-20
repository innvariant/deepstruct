import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from deepstruct.graph import CachedLayeredGraph
from deepstruct.graph import LayeredGraph


class DeepCellDAN(nn.Module):
    def __init__(
        self, num_classes, input_channel_size, cell_constructor, structure: LayeredGraph
    ):
        super(DeepCellDAN, self).__init__()

        self._structure = structure
        assert structure.num_layers > 0
        self._num_classes = num_classes
        self._input_channel_size = input_channel_size

        self._input_nodes = [n for n in structure.nodes if structure.in_degree(n) == 0]
        self._output_nodes = [
            n for n in structure.nodes if structure.out_degree(n) == 0
        ]
        assert len(self._input_nodes) > 0, "No input nodes in structure: len=%d" % len(
            self.input_nodes
        )
        assert (
            len(self._output_nodes) > 0
        ), "No output nodes in structure: len=%d" % len(self.output_nodes)

        # Each cell has to have output channel size of 1
        # The input channel size of a cell is its in_degree
        # Except for input nodes, then the channel size equals the input channel size
        self._node_cells = {
            node: cell_constructor(
                is_input=node in self._input_nodes,
                is_output=node in self._output_nodes,
                in_degree=structure.in_degree(node),
                out_degree=structure.out_degree(node),
                layer=structure.get_layer(node),
                input_channel_size=self._input_channel_size,
            )
            for node in self._structure.nodes
        }
        self._nodes = nn.ModuleList(list(self._node_cells.values()))

        self._output_channel_size = 100
        self._last_convolution = nn.Conv2d(
            len(self._output_nodes), self._output_channel_size, (1, 1)
        )
        self._last_batch_norm = nn.BatchNorm2d(self._output_channel_size)
        self._last_fc = nn.Linear(self._output_channel_size, self._num_classes)

    def forward(self, input):
        # input : [batch_size, input_channel_size, ?, ..], e.g. [100, 1, 28, 28] or [100, 3, 32, 32]

        # x = input.flatten(start_dim=1)
        # x: [batch_size, ?*?*..], e.g. [100, 784] or [100, 3072]

        outputs = {
            input_node: self._node_cells[input_node](input)
            for input_node in self._input_nodes
        }
        for node in nx.topological_sort(self._structure):
            # Skip input nodes, as we already processed them
            if node in self._input_nodes:
                continue

            # Collect all inputs of incoming edges and compute the maximum height and width (last two dimensions) of
            # all possible shapes of those tensors
            inputs_raw = []
            max_h = 0
            max_w = 0
            for (u, v) in self._structure.in_edges(node):
                inputs_raw.append(outputs[u])
                max_h = max(max_h, outputs[u].shape[2])
                max_w = max(max_w, outputs[u].shape[3])

            input_stack = []
            for cell_output in inputs_raw:
                # input can be [batch_size, 1, 20?, 20?] or [batch_size, 1, 30?, 20?]
                # it is then zero padded to [batch_size, 1, max(?), max(?)]
                zero_padded_input = torch.zeros(
                    cell_output.shape[0],
                    cell_output.shape[1],
                    max_h,
                    max_w,
                    device=cell_output.device,
                )
                zero_padded_input[
                    :, :, : cell_output.shape[2], : cell_output.shape[3]
                ] = cell_output
                input_stack.append(zero_padded_input)
            # The current input is then concatenated along the channel dimension to build up to
            # current_input: [batch_size, #in_edges(node), max(?), max(?)]
            current_input = torch.cat(input_stack, 1)

            outputs[node] = self._node_cells[node](current_input)

        # Same trick again: compute the maximum height- and width-dimension of all output tensors of nodes which are
        # output nodes. We have to iterate through all of those nodes.
        max_h = 0
        max_w = 0
        for output_node in self._output_nodes:
            max_h = max(max_h, outputs[output_node].shape[2])
            max_w = max(max_w, outputs[output_node].shape[3])

        # Now we obtained the maximum width and height, so we can zero-pad all tensors to that dimension
        final_output_stack = []
        for output_node in self._output_nodes:
            current_output = outputs[output_node]
            # current_output can be [batch_size, 1, 20?, 20?] or [batch_size, 1, 30?, 20?]
            # it is then zero padded to [batch_size, 1, max(?), max(?)]
            zero_padded = torch.zeros(
                current_output.shape[0],
                current_output.shape[1],
                max_h,
                max_w,
                device=current_output.device,
            )
            zero_padded[
                :, :, : current_output.shape[2], : current_output.shape[3]
            ] = current_output
            final_output_stack.append(zero_padded)
        final_output = torch.cat(
            final_output_stack, 1
        )  # which is now [batch_size, #output_nodes, max(?), max(?)]

        # Now apply a last convolution to have a fixed output channel size
        final_output = self._last_convolution(
            final_output
        )  # convolved down to [batch_size, self._output_channel_size, ?, ?]
        final_output = self._last_batch_norm(
            final_output
        )  # [batch_size, self._output_channel_size, ?, ?]
        # And apply pooling, so we know the exact number of final features for a linear classifier
        final_output = torch.nn.functional.adaptive_avg_pool2d(
            final_output, (1, 1)
        )  # [batch_size, self._output_channel_size, 1, 1]
        final_output = final_output.view(
            final_output.size(0), -1
        )  # [batch_size, self._output_channel_size]
        return self._last_fc(final_output)  # [batch_size, _num_classes]


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

    def __init__(self, input_size, num_classes, structure: LayeredGraph):
        super(MaskedDeepDAN, self).__init__()

        self._structure = structure
        assert structure.num_layers > 0

        # Multiple dimensions for input size are flattened out
        if type(input_size) is tuple or type(input_size) is torch.Size:
            input_size = np.prod(input_size)
        input_size = int(input_size)

        self.layer_first = MaskedLinearLayer(input_size, structure.first_layer_size)
        self.layers_main_hidden = nn.ModuleList(
            [
                MaskedLinearLayer(
                    structure.get_layer_size(cur_lay - 1),
                    structure.get_layer_size(cur_lay),
                )
                for cur_lay in structure.layers[1:]
            ]
        )

        for layer_idx, layer in zip(structure.layers[1:], self.layers_main_hidden):
            mask = torch.zeros(
                structure.get_layer_size(layer_idx),
                structure.get_layer_size(layer_idx - 1),
            )
            for source_idx, source_vertex in enumerate(
                structure.get_vertices(layer_idx - 1)
            ):
                for target_idx, target_vertex in enumerate(
                    structure.get_vertices(layer_idx)
                ):
                    if structure.has_edge(source_vertex, target_vertex):
                        mask[target_idx][source_idx] = 1
            layer.set_mask(mask)

        skip_layers = []
        self._skip_targets = {}
        for target_layer in structure.layers[2:]:
            target_size = structure.get_layer_size(target_layer)
            for distant_source_layer in structure.layers[: target_layer - 1]:
                if structure.layer_connected(distant_source_layer, target_layer):
                    if target_layer not in self._skip_targets:
                        self._skip_targets[target_layer] = []

                    skip_layer = MaskedLinearLayer(
                        structure.get_layer_size(distant_source_layer), target_size
                    )
                    mask = torch.zeros(
                        structure.get_layer_size(target_layer),
                        structure.get_layer_size(distant_source_layer),
                    )
                    for source_idx, source_vertex in enumerate(
                        structure.get_vertices(distant_source_layer)
                    ):
                        for target_idx, target_vertex in enumerate(
                            structure.get_vertices(target_layer)
                        ):
                            if structure.has_edge(source_vertex, target_vertex):
                                mask[target_idx][source_idx] = 1
                    skip_layer.set_mask(mask)

                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append(
                        {"layer": skip_layer, "source": distant_source_layer}
                    )
        self.layers_skip_hidden = nn.ModuleList(skip_layers)

        self.layer_out = MaskedLinearLayer(structure.last_layer_size, num_classes)
        self.activation = nn.ReLU()

    def generate_structure(self, include_input=False, include_output=False):
        structure = CachedLayeredGraph()

        node_number_offset = 0
        layer_nodeidx2node = {}

        if include_input:
            layer = self.layer_first
            layer_nodeidx2node[0] = {
                node_idx: node
                for node_idx, node in enumerate(
                    np.arange(
                        node_number_offset, node_number_offset + layer.mask.shape[1]
                    )
                )
            }
            node_number_offset += layer.mask.shape[1]
            structure.add_nodes_from(layer_nodeidx2node[0].values())

            for source_node_idx in range(layer.mask.shape[1]):
                source_node = layer_nodeidx2node[0][source_node_idx]

                layer_nodeidx2node[1] = {
                    node_idx: node
                    for node_idx, node in enumerate(
                        np.arange(
                            node_number_offset, node_number_offset + layer.mask.shape[0]
                        )
                    )
                }
                node_number_offset += layer.mask.shape[0]
                structure.add_nodes_from(layer_nodeidx2node[1].values())

                for target_node_idx in range(layer.mask.shape[0]):
                    if layer.mask[target_node_idx][source_node_idx]:
                        target_node = layer_nodeidx2node[1][target_node_idx]
                        structure.add_edge(source_node, target_node)

        # Main hidden layers
        for source_layer_idx, layer in enumerate(self.layers_main_hidden, start=1):
            if source_layer_idx not in layer_nodeidx2node:
                layer_nodeidx2node[source_layer_idx] = {
                    node_idx: node
                    for node_idx, node in enumerate(
                        np.arange(
                            node_number_offset, node_number_offset + layer.mask.shape[1]
                        )
                    )
                }
                node_number_offset += layer.mask.shape[1]
                structure.add_nodes_from(layer_nodeidx2node[source_layer_idx].values())

            target_layer_idx = source_layer_idx + 1
            if target_layer_idx not in layer_nodeidx2node:
                layer_nodeidx2node[target_layer_idx] = {
                    node_idx: node
                    for node_idx, node in enumerate(
                        np.arange(
                            node_number_offset, node_number_offset + layer.mask.shape[0]
                        )
                    )
                }
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
                source_layer = target["layer"]
                # We have to shift target layer index as layer_nodeidx2node is indexed differently
                target_layer_idx = target_idx + 1
                # Also shift the source layer index by one as we indexed layer_nodeidx2node differently
                source_layer_idx = target["source"] + 1

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
            layer_nodeidx2node[layer_indices] = {
                node_idx: node
                for node_idx, node in enumerate(
                    np.arange(
                        node_number_offset, node_number_offset + layer.mask.shape[0]
                    )
                )
            }
            node_number_offset += layer.mask.shape[0]
            structure.add_nodes_from(layer_nodeidx2node[layer_indices].values())

            for source_node_idx in range(layer.mask.shape[1]):
                source_node = layer_nodeidx2node[layer_indices - 1][source_node_idx]
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
        for layer, layer_idx in zip(
            self.layers_main_hidden, self._structure.layers[1:]
        ):
            out = self.activation(layer(last_output))

            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target["layer"]
                    source_idx = skip_target["source"]

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

    def __init__(self, input_size, num_classes, hidden_layers: list):
        super(MaskedDeepFFN, self).__init__()
        assert len(hidden_layers) > 0

        # Multiple dimensions for input size are flattened out
        if type(input_size) is tuple or type(input_size) is torch.Size:
            input_size = np.prod(input_size)
        input_size = int(input_size)

        self.layer_first = MaskedLinearLayer(input_size, hidden_layers[0])
        self.layers_hidden = nn.ModuleList(
            [
                MaskedLinearLayer(hidden_layers[lay], hid)
                for lay, hid in enumerate(hidden_layers[1:])
            ]
        )
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
                for target_node_idx, target_node in enumerate(
                    range(layer.mask.shape[0])
                ):
                    if layer.mask[target_node_idx][source_node_idx]:
                        structure.add_edge(
                            offset_source + source_node, offset_target + target_node
                        )
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
        out = self.activation(
            self.layer_first(input.flatten(start_dim=1))
        )  # [B, n_hidden_1]
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

        self.register_buffer(
            "mask", torch.ones((out_features, in_feature), dtype=torch.bool)
        )
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
        self.mask = torch.ones(
            self.weight.shape, dtype=torch.bool, device=self.mask.device
        )
        self.mask[torch.where(abs(self.weight) < theta)] = False

    def get_weight(self):
        return self.weight

    def reset_parameters(self, keep_mask=False):
        super().reset_parameters()
        # hasattr() is necessary because reset_parameters() is called in __init__ of Linear(), but buffer 'mask'
        # may only be registered after super() call, thus 'mask' might not be defined as buffer / attribute, yet
        if hasattr(self, "mask") and not keep_mask:
            self.mask = torch.ones(self.weight.size(), dtype=torch.bool)

    def forward(self, input):
        x = (
            input.float()
        )  # In case we get a double-tensor passed, force it to be float for multiplications to work

        # Possibly store the layer input
        if self.keep_layer_input:
            self.layer_input = x.data

        return F.linear(x, self.weight * self.mask, self.bias)
