from abc import abstractmethod

import numpy as np
import torch.nn

from deepstruct.topologie_representation import LayeredFXGraph


class NodeMapper:

    @abstractmethod
    def add_node(self, graph, predecessors, **kwargs):
        pass


class CustomNodeMap:

    def __init__(self, node_mappers: dict, default_mapper: NodeMapper):
        self.node_mappers = node_mappers
        self.default_mapper = default_mapper

    def map_node(self, graph, module, predecessors, **kwargs):
        kwargs['module'] = module
        self.node_mappers.get(module, self.default_mapper).add_node(graph, predecessors, **kwargs)


class HighLevelNodeMap(CustomNodeMap):

    def __init__(self):
        super().__init__({}, All2VertexNodeMapper())


class LowLevelNodeMap(CustomNodeMap):

    def __init__(self, threshold=-1):
        super().__init__({
            torch.nn.Linear: Linear2LayerMapper(threshold),
            torch.nn.Conv2d: Conv2LayerMapper(threshold)
        }, All2VertexNodeMapper())


class All2VertexNodeMapper(NodeMapper):

    def add_node(self, graph: LayeredFXGraph, predecessors, **kwargs):
        node_name = kwargs.pop('name')
        output_layer_len = 0
        for pred in predecessors:
            output_layer_len += graph.get_output_layer_len(str(pred))
        if output_layer_len == 0:
            graph.add_vertices(1, node_name, **kwargs)
        else:
            graph.add_vertices(output_layer_len, node_name, **kwargs)
        graph.add_edges(predecessors, node_name)


class Linear2LayerMapper(NodeMapper):

    def __init__(self, threshold):
        self.threshold = threshold

    def add_node(self, graph: LayeredFXGraph, predecessors, **kwargs):
        model = kwargs.get("origin_module")
        in_features = model.in_features
        out_features = model.out_features
        mask = torch.ones((out_features, in_features), dtype=torch.bool)
        mask[torch.where(abs(model.weight) < self.threshold)] = False  # L1-Pruning
        kwargs['mask'] = mask
        node_name = kwargs.pop('name')
        graph.add_vertices(in_features, node_name, output_layer_size=out_features, **kwargs)
        graph.add_edges(predecessors, node_name)


class Conv2LayerMapper(NodeMapper):

    def __init__(self, threshold):
        self.threshold = threshold

    def add_node(self, graph: LayeredFXGraph, predecessors, **kwargs):
        model = kwargs.get('origin_module')
        shape = kwargs.get('shape')
        node_name = kwargs.pop('name')
        channels_in = model.in_channels
        channels_out = model.out_channels
        size_kernel = model.kernel_size
        stride = model.stride
        padding = model.padding

        def input_shape(size, dim):
            return int(
                (size - 1) * stride[dim] - 2 * padding[dim] + size_kernel[dim]
            )

        output_width = shape[-1]
        output_height = shape[-2]
        input_height = input_shape(output_height, 0)
        input_width = input_shape(output_width, 1)
        input_neurons_count = channels_in * input_width * input_height
        output_neurons_count = channels_out * output_height * output_width
        input_neurons = graph.add_vertices(input_neurons_count,
                                           node_name,
                                           output_layer_size=output_neurons_count,
                                           **kwargs
                                           )
        output_neurons = [input_neurons[-1] + i + 1 for i in range(output_neurons_count)]

        def get_input_neuron(channel: int, row: int, col: int):
            return int(input_neurons[int((col * input_height + row) + (channel * input_width * input_height))])

        def get_output_neuron(channel_out: int, row: int, col: int):
            return int(output_neurons[int((col * output_height + row) + (channel_out * output_width * output_height))])

        edges = []
        for idx_channel_out in range(channels_out):
            for idx_channel_in in range(channels_in):
                out_col = 0
                offset_height = -padding[0]
                while offset_height + size_kernel[0] <= input_height:
                    out_row = 0
                    offset_width = -padding[1]
                    while offset_width + size_kernel[1] <= input_width:
                        target = get_output_neuron(idx_channel_out, out_row, out_col)
                        for col in range(max(0, offset_height), min(offset_height + size_kernel[0], input_height), ):
                            for row in range(max(0, offset_width), min(offset_width + size_kernel[1], input_width), ):
                                source = get_input_neuron(idx_channel_in, row, col)
                                edges.append((source, target))
                        offset_width += stride[1]
                        out_row += 1
                    offset_height += stride[0]
                    out_col += 1
        graph.edges_for_name[node_name] = edges
        graph.add_edges(predecessors, node_name)
