from abc import abstractmethod

import numpy as np
import torch.nn


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
        }, All2VertexNodeMapper())


class All2VertexNodeMapper(NodeMapper):

    def add_node(self, graph, predecessors, **kwargs):
        node_name = kwargs.pop('name')
        mask_len = 0
        for pred in predecessors:
            mask_len += graph.get_mask_len(str(pred))
        if mask_len == 0:
            graph.add_vertex(node_name, **kwargs)
        else:
            for _ in range(mask_len):
                graph.add_vertex(node_name, **kwargs)
        graph.add_edges(predecessors, node_name)


class Linear2LayerMapper(NodeMapper):

    def __init__(self, threshold):
        self.threshold = threshold

    def add_node(self, graph, predecessors, **kwargs):
        model = kwargs.get("origin_module")
        in_features = model.in_features
        out_features = model.out_features
        mask = torch.ones((out_features, in_features), dtype=torch.bool)
        mask[torch.where(abs(model.weight) < self.threshold)] = False
        node_name = kwargs.pop('name')
        for i in range(in_features):
            kwargs['mask'] = mask[:, i]
            graph.add_vertex(node_name, **kwargs)
        graph.add_edges(predecessors, node_name)


class Conv2LayerMapper(NodeMapper):

    def __init__(self, threshold):
        self.threshold = threshold

    def add_node(self, graph, predecessors, **kwargs):
        model = kwargs.get('origin_module')
        shape = kwargs.get('shape')

        channels_in = model.in_channels
        channels_out = model.out_channels
        size_kernel = model.kernel_size
        stride = model.stride
        padding = model.padding
        width = shape[-1]
        height = shape[-2]
        input_neurons_count = channels_in * width * height
        input_neurons = []
        output_neurons = []

        def output_shape(size, dim):
            return int(
                np.floor((size - size_kernel[dim] + 2 * padding[dim]) / stride[dim]) + 1
            )

        output_height = output_shape(height, 0)
        output_width = output_shape(width, 1)
        output_neurons_count = channels_out * output_height * output_width

        def get_input_neuron(channel: int, row: int, col: int):
            return int(input_neurons[int((col * height + row) + (channel * width * height))])

        def get_output_neuron(channel_out: int, row: int, col: int):
            return int(output_neurons[int((col * output_height + row) + (channel_out * output_width * output_height))])

        for idx_channel_out in range(channels_out):
            for idx_channel_in in range(channels_in):
                out_col = 0
                offset_height = -padding[0]
                while offset_height + size_kernel[0] <= height:
                    out_row = 0
                    offset_width = -padding[1]
                    while offset_width + size_kernel[1] <= width:
                        target = get_output_neuron(idx_channel_out, out_row, out_col)
                        edges = []
                        for col in range(
                                max(0, offset_height),
                                min(offset_height + size_kernel[0], height),
                        ):
                            for row in range(
                                    max(0, offset_width),
                                    min(offset_width + size_kernel[1], width),
                            ):
                                source = get_input_neuron(idx_channel_in, row, col)
                                edges.append((source, target))
                        graph.add_edges_from(edges)
                        offset_width += stride[1]
                        out_row += 1
                    offset_height += stride[0]
                    out_col += 1

    def _add_node(self, graph, predecessors, **kwargs):
        node_name = kwargs.pop('name')
        graph.add_vertex(node_name, **kwargs)
        graph.add_edges(predecessors, node_name)
