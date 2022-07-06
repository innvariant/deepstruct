from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deprecated import deprecated
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
    def apply_mask(self) -> MaskableModule:
        for layer in maskable_layers(self):
            layer.apply_mask()
        return self

    def recompute_mask(self, theta=0.0001) -> MaskableModule:
        for layer in maskable_layers(self):
            layer.recompute_mask(theta)
        return self

    def reset_parameters(self, keep_mask=False) -> MaskableModule:
        super().reset_parameters()
        for layer in maskable_layers(self):
            layer.recompute_mask(keep_mask=keep_mask)
        return self

    @property
    def maskable_children(self):
        return maskable_layers(self)

    @property
    def maskable_children_with_name(self):
        return maskable_layers_with_name(self)


class MaskedDeepDAN(MaskableModule):  # nn.Module
    """
    A deep directed acyclic network model which is capable of masked layers and masked skip-layer connections.
    """

    def __init__(
        self,
        size_input,
        size_output: int,
        structure: LayeredGraph,
        use_layer_norm: bool = False,
    ):
        super(MaskedDeepDAN, self).__init__()

        self._structure = structure
        self._use_layer_norm = True if use_layer_norm else False
        assert structure.num_layers > 0

        # Multiple dimensions for input size are flattened out
        if type(size_input) is tuple or type(size_input) is torch.Size:
            size_input = np.prod(size_input)
        size_input = int(size_input)

        self.layer_first = MaskedLinearLayer(size_input, structure.first_layer_size)
        self.layers_main_hidden = nn.ModuleList(
            [
                MaskedLinearLayer(
                    structure.get_layer_size(cur_lay - 1),
                    structure.get_layer_size(cur_lay),
                )
                for cur_lay in structure.layers[1:]
            ]
        )

        self._layers_normalization = []
        if self._use_layer_norm:
            self._layers_normalization.append(nn.LayerNorm(structure.get_layer_size(0)))

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
            layer.mask = mask

            if self._use_layer_norm:
                self._layers_normalization.append(
                    nn.LayerNorm(structure.get_layer_size(layer_idx))
                )

        self.layer_normalizations = nn.ModuleList(self._layers_normalization)

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
                    skip_layer.mask = mask

                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append(
                        {"layer": skip_layer, "source": distant_source_layer}
                    )
        self.layers_skip_hidden = nn.ModuleList(skip_layers)

        self.layer_out = MaskedLinearLayer(structure.last_layer_size, size_output)
        self.activation = nn.ReLU()

    def forward(self, input):
        # input : [batch_size, ?, ?, ..], e.g. [100, 1, 28, 28] or [100, 3, 32, 32]
        x = input.flatten(start_dim=1)
        # x: [batch_size, ?*?*..], e.g. [100, 784] or [100, 3072]
        last_output = self.activation(self.layer_first(x))
        layer_results = {0: last_output}
        for layer, layer_idx in zip(
            self.layers_main_hidden, self._structure.layers[1:]
        ):
            # Apply directly layer transformation
            out_layer = layer(last_output)
            if self._use_layer_norm:
                out_layer = self._layers_normalization[layer_idx](out_layer)
            out = self.activation(out_layer)

            # Sum up all additional layer transformations from skip-connections
            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target["layer"]
                    source_idx = skip_target["source"]

                    out += self.activation(source_layer(layer_results[source_idx]))

            layer_results[layer_idx] = out  # copy?
            last_output = out

        return self.layer_out(last_output)

    @deprecated(
        reason="Generating a modules graph structure will be handled from outside through a functor object.",
        version="0.8.0",
    )
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


class MaskedDeepFFN(MaskableModule):
    """
    A deep feed-forward network model which is capable of masked layers.
    Masked layers can represent sparse structures between consecutive layers.
    This representation is suitable for feed-forward sparse networks, probably with density 0.5 and above per layer.
    """

    def __init__(
        self, size_input, size_output, hidden_layers: list, use_layer_norm: bool = False
    ) -> MaskedDeepFFN:
        super(MaskedDeepFFN, self).__init__()
        assert len(hidden_layers) > 0
        self._activation = nn.ReLU()

        # Multiple dimensions for input size are flattened out
        if type(size_input) is tuple or type(size_input) is torch.Size:
            size_input = np.prod(size_input)
        size_input = int(size_input)

        self._layer_first = MaskedLinearLayer(size_input, hidden_layers[0])
        self._layers_hidden = torch.nn.ModuleList()
        for cur_lay, size_h in enumerate(hidden_layers[1:]):
            self._layers_hidden.append(
                MaskedLinearLayer(hidden_layers[cur_lay], size_h)
            )

            if use_layer_norm:
                self._layers_hidden.append(torch.nn.LayerNorm(size_h))

            self._layers_hidden.append(self._activation)

        self._layer_out = MaskedLinearLayer(hidden_layers[-1], size_output)

    @deprecated(
        reason="Generating a modules graph structure will be handled from outside through a functor object.",
        version="0.8.0",
    )
    def generate_structure(self, include_input=False, include_output=False):
        structure = CachedLayeredGraph()

        def add_edges(structure, layer, offset_source):
            if not isinstance(layer, MaskedLinearLayer):
                return offset_source
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
            offset_source = add_edges(structure, self._layer_first, 0)

        for layer in self._layers_hidden:
            offset_source = add_edges(structure, layer, offset_source)

        if include_output:
            add_edges(structure, self._layer_out, offset_source)

        return structure

    def forward(self, input):
        # input : [batch_size, ?, ?, ..], e.g. [100, 1, 28, 28] or [100, 3, 32, 32]
        out = self._activation(
            self._layer_first(input.flatten(start_dim=1))
        )  # [B, n_hidden_1]
        for layer in self._layers_hidden:
            out = layer(out)
        return self._layer_out(out)  # [B, n_out]


def maskable_layers(model: nn.Module) -> Iterable[MaskableModule]:
    for child in model.children():
        if type(child) is MaskedLinearLayer:
            yield child
        elif type(child) is nn.ModuleList:
            for layer in maskable_layers(child):
                yield layer


def maskable_layers_with_name(model: nn.Module) -> Iterable[MaskableModule]:
    for name, child in model.named_children():
        if type(child) is MaskedLinearLayer:
            yield name, child
        elif type(child) is nn.ModuleList:
            for name, layer in maskable_layers_with_name(child):
                yield name, layer


@deprecated(
    reason="Redundant function name. Should simply use maskable_layers", version="0.9.0"
)
def prunable_layers(network) -> Iterable[MaskableModule]:
    return maskable_layers(network)


@deprecated(
    reason="Redundant function name. Should simply use maskable_layers_with_name",
    version="0.9.0",
)
def prunable_layers_with_name(network) -> Iterable[MaskableModule]:
    return maskable_layers_with_name(network)


class MaskedLinearLayer(nn.Linear, MaskableModule):
    def __init__(
        self,
        in_feature,
        out_features,
        bias=True,
        keep_layer_input=False,
        mask_as_params: bool = False,
    ):
        """
        :param in_feature:          The number of features that are inserted in the layer.
        :param out_features:        The number of features that are returned by the layer.
        :param bias:                Iff each neuron in the layer should have a bias unit as well.
        """

        self._masks_as_params = True if mask_as_params else False
        self._saliency = None
        super().__init__(in_feature, out_features, bias)

        if mask_as_params:
            # Mask as a parameter could still be updated through optimization, e.g. by momentum
            # For the purpose of not considering them in optimization, you can set requires_grad=False on them
            # self._mask = nn.Parameter(torch.ones((out_features, in_feature, 2), dtype=torch.float32), requires_grad=False)
            # self._mask = nn.Parameter(torch.Tensor(out_features, in_feature, 2))
            self._mask = nn.Parameter(
                torch.ones((out_features, in_feature, 2), dtype=torch.float32)
            )
        elif not mask_as_params:
            # Masks as buffers are considered in persistence, putting the computation to GPU or changing its types
            # but are not contained in the set of parameters for optimization
            self.register_buffer(
                "_mask", torch.ones((out_features, in_feature), dtype=torch.bool)
            )
        self.keep_layer_input = keep_layer_input
        self.layer_input = None

    def reset_parameters(self, keep_mask=False):
        super().reset_parameters()
        if hasattr(self, "_mask") and self._masks_as_params and not keep_mask:
            # self.mask = torch.round(torch.rand_like(self._mask))
            self.mask = torch.ones_like(self.weight)
        elif hasattr(self, "_mask") and not keep_mask:
            # hasattr() is necessary because reset_parameters() is called in __init__ of Linear(), but buffer 'mask'
            # may only be registered after super() call, thus 'mask' might not be defined as buffer / attribute, yet
            self.mask = torch.ones(self.weight.size(), dtype=torch.bool)

    @property
    def mask(self):
        return (
            self._mask
            if not self._masks_as_params
            else torch.argmax(torch.softmax(self._mask, dim=2), dim=2)
        )

    @mask.setter
    def mask(self, mask):
        """
        :param mask: Binary mask of shape (out_feature, in_feature)
        :return:
        """
        if self._masks_as_params:
            # print("before setting", self._mask)
            mask_inverted = 1 - mask
            alphas = torch.zeros_like(self._mask, dtype=torch.float32)
            alphas[:, :, 0] = mask_inverted * 0.9 + mask * 0.1
            alphas[:, :, 1] = mask * 0.9 + mask_inverted * 0.1
            # self._mask[:, :, 0] = mask_inverted * 0.9 + mask * 0.1
            # self._mask[:, :, 1] = mask * 0.9 + mask_inverted * 0.1
            self._mask = nn.Parameter(alphas)
            # print("after setting", self._mask)
        else:
            self._mask = mask

    def __getitem__(self, key):
        return (
            self._mask[key]
            if not self._masks_as_params
            else torch.argmax(torch.softmax(self._mask[key], dim=1), dim=1)
        )

    def __setitem__(self, key, value):
        with torch.no_grad():
            if self._masks_as_params:
                mask_inverted = 1 - value
                alphas = torch.zeros_like(self._mask[key])
                alphas[0] = mask_inverted * 0.9 + value * 0.1
                alphas[1] = value * 0.9 + mask_inverted * 0.1
                self._mask[key].copy_(alphas)
            else:
                self._mask[key].copy_(torch.tensor(value))
            # self._mask[key] = value

    @deprecated(
        reason="Accessing mask should be done via the property accessor .mask",
        version="0.8.0",
    )
    def get_mask(self):
        return self.mask

    @deprecated(
        reason="Accessing mask should be done via the property accessor .mask",
        version="0.8.0",
    )
    def set_mask(self, mask):
        self.mask = Variable(mask)

    def get_weight_count(self):
        return self.mask.sum()

    def apply_mask(self):
        # Assigning "self.weight = torch.nn.Parameter(self.weight.mul(self.mask))" might have side effects
        # Using direct manipulation on tensor "self.weight.data"
        self.weight.data = self.weight * self.mask

    @property
    def saliency(self):
        if self._saliency is None:
            return self.weight.data.abs()
        else:
            return self._saliency

    @saliency.setter
    def saliency(self, saliency):
        if saliency.size() != self.weight.size():
            raise ValueError(
                "The provided saliency measure for this layer must be of same shape as the weights."
            )

        self._saliency = saliency

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

    @deprecated(
        reason="Unnecessary additional accessor for the modules weight. Use the property accessor '.weight' of nn.Linear",
        version="0.8.0",
    )
    def get_weight(self):
        return self.weight

    def forward(self, input):
        x = (
            input.float()
        )  # In case we get a double-tensor passed, force it to be float for multiplications to work

        # Possibly store the layer input
        if self.keep_layer_input:
            self.layer_input = x.data

        return F.linear(x, self.weight * self.mask, self.bias)
