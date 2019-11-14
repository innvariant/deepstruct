import torch
import torch.nn as nn
from typing import Callable
from torch.nn import functional as F

from pypaddle.sparse import LayeredGraph, MaskedLinearLayer


class LinearBlockUnit(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super(LinearBlockUnit, self).__init__()
        self.act = nn.ReLU()
        #self.conv = nn.Conv2d(1, 10, kernel_size=3, stride=stride, padding=1)
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        out = self.act(x)
        out = self.linear(out)
        out = self.bn(out)
        return out


class FiveWaySparseBlockNet(nn.Module):
    def __init__(self, input_size, output_classes, hidden_block_size=100):
        super(FiveWaySparseBlockNet, self).__init__()

        self.input_blocks = nn.ModuleList()
        for _ in range(5):
            self.input_blocks.append(LinearBlockUnit(input_size, hidden_block_size))

        self.hidden_blocks = nn.ModuleList()
        for _ in range(5):
            self.hidden_blocks.append(LinearBlockUnit(hidden_block_size, hidden_block_size))

        self.output_blocks = nn.ModuleList()
        for _ in range(5):
            self.output_blocks.append(LinearBlockUnit(hidden_block_size, output_classes))

    def forward(self, x):
        input_to_blocks = {}
        for idx, block_unit in enumerate(self.input_blocks):
            input_to_blocks[idx] = block_unit(x)

        hidden_block_results = {}
        for idx, block_unit in enumerate(self.hidden_blocks):
            hidden_block_results[idx] = block_unit(input_to_blocks[idx])

        output_blocks = []
        for idx, block_unit in enumerate(self.output_blocks):
            output_blocks.append(block_unit(hidden_block_results[idx]))

        return torch.mean(torch.stack(output_blocks), dim=0)



cell_constructor_func: Callable[[bool, bool, int, int, int], nn.Module] = lambda is_input, is_output, in_degree, out_degree, layer: nn.Conv2d(10, 10, 1)
skip_map_fn: Callable[[int, int], nn.Module] = lambda layer: nn.Conv2d(10, 10, 1)
layer_channel_size_func: Callable[[int], int] = lambda layer: 10


class DeepDACellNetwork(nn.Module):
    def __init__(self, num_classes, cell_constructor: cell_constructor_func, layer_channel_size: layer_channel_size_func, fn_skip_map: skip_map_fn, structure: LayeredGraph):
        super(DeepDACellNetwork, self).__init__()

        self._structure = structure
        assert structure.num_layers > 0
        self._layer_channel_size = layer_channel_size

        self._input_nodes = [n for n in structure.nodes if structure.in_degree(n) == 0]
        self._output_nodes = [n for n in structure.nodes if structure.out_degree(n) == 0]
        assert len(self._input_nodes) > 0, 'No input nodes in structure: len=%d' % len(self.input_nodes)
        assert len(self._output_nodes) > 0, 'No output nodes in structure: len=%d' % len(self.output_nodes)

        # For each node in our structure, we create a cell based on the given cell constructor
        # The cell constructor has to return a nn.Module which maps [B, C_s, N, M, D] -> [B, C_t, N, M]
        # in which B stands for batch size, C_s for the channel size of the source layer defined by
        # layer_channel_size_func, N and M for width and height of the features, C_t for the channel size of the target
        # layer defined by layer_channel_size_func and D for the outgoing degree of the particular node (which needs to
        # be condensed, e.g. by aggregating on the last dimension)
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

        # Additionaly maps for skip-connections are created. These map channels from one
        self._node_maps = {}
        self._node_maps_list = []
        for source_layer in self._structure.layers[:-1]:
            self._node_maps[source_layer] = {}
            source_output_channel_size = self._layer_channel_size(source_layer+1)
            for target_layer in self._structure.layers[source_layer+1:]:
                target_channel_size = self._layer_channel_size(target_layer+1)
                if source_output_channel_size is not target_channel_size:
                    #self._node_maps[source_layer][target_layer] = nn.Conv2d(source_output_channel_size, target_channel_size, kernel_size=1)
                    self._node_maps[source_layer][target_layer] = fn_skip_map(source_output_channel_size, target_channel_size)
                    #print(self._node_maps[source_layer][target_layer])
                    self._node_maps_list.append(self._node_maps[source_layer][target_layer])
        self._maps = nn.ModuleList(self._node_maps_list)

        # Extract number of out channels from node of --last-- layer (not any output node)
        # A node in the last layer has actually the number of out channels everything is mapped to
        any_output_node = self._structure.get_vertices(self._structure.last_layer)[0]
        num_out_channel = self._node_cells[any_output_node].num_out_channel

        self.last_representation_size = 1000
        self.convlast = nn.Conv2d(num_out_channel, self.last_representation_size, kernel_size=1)
        self.bnlast = nn.BatchNorm2d(self.last_representation_size)
        self.fc = MaskedLinearLayer(self.last_representation_size, num_classes)

    def _apply_skip_layer_adaption(self, source_node, target_node, input):
        source_layer = self._structure.get_layer(source_node)
        target_layer = self._structure.get_layer(target_node)
        source_size = self._layer_channel_size(source_layer)
        target_size = self._layer_channel_size(target_layer)

    def forward(self, input):
        # input: [batch_size, channels, N, M] = [B, C_in, N, M]

        outputs = { node: None for node in self._structure.nodes }

        for layer in self._structure.layers:
            for current_node in self._structure.get_vertices(layer):

                if current_node in self._input_nodes:
                    current_input = [input]
                else:
                    #current_input = [ self._node_maps[self._structure.get_layer(u)][layer](outputs[u])
                    #                  if layer in self._node_maps[self._structure.get_layer(u)] else outputs[u] for (u, _) in self._structure.in_edges(current_node)]
                    current_input = []
                    for (source, _) in self._structure.in_edges(current_node):
                        source_layer = self._structure.get_layer(source)
                        if layer in self._node_maps[source_layer]:
                            """print('From source layer', source_layer)
                            print('To target layer', layer)
                            print(self._node_maps[source_layer][layer].in_channels)
                            print(self._node_maps[source_layer][layer].out_channels)
                            print('Source output shape:', outputs[source].shape)
                            print('Mapped source output to new channel', self._node_maps[source_layer][layer].out_channels)"""
                            current_input.append(self._node_maps[source_layer][layer](outputs[source]))
                        else:
                            current_input.append(outputs[source])

                feed = torch.stack(current_input, dim=-1)

                outputs[current_node] = self._node_cells[current_node](feed)

            # Force GPU to possibly free memory as stacks of collected inputs/outputs can grow rapidly
            torch.cuda.empty_cache()

        # Collect all outputs of the output nodes into a list and stack it
        # Use the channel output size of the last layer as targeted outgoing channel size
        # Every output node from previous layers are mapped through a convolution to the next layer
        out_list = []
        for node in self._output_nodes:
            source_layer = self._structure.get_layer(node)
            if source_layer is self._structure.last_layer or self._structure.last_layer not in self._node_maps[source_layer]:
                if source_layer is not self._structure.last_layer:
                    print('Channel size source layer:', self._layer_channel_size(source_layer))
                    print('Channel size last layer:', self._layer_channel_size(self._structure.last_layer))
                    print('Output shape:', outputs[node].shape)
                out_list.append(outputs[node])
            else:
                map_to_last_layer = self._node_maps[source_layer][self._structure.last_layer]
                out_list.append(map_to_last_layer(outputs[node]))
        print([o.shape for o in out_list])
        result = torch.mean(torch.stack(out_list), dim=0)  # [B, C_out, N, M]

        # Classifier
        y = F.relu(result) # [B, C_out, N, M]
        y = self.convlast(y) # [B, 1000, N, M]
        y = self.bnlast(y) # [B, 1000, N, M]
        y = F.adaptive_avg_pool2d(y, (1, 1)) # [B, 1000, 1, 1]
        y = y.view(y.size(0), -1) # [B, 1000]
        return self.fc(y) # [B, num_classes]
        #return F.log_softmax(y, dim=1) # [B, num_classes]
