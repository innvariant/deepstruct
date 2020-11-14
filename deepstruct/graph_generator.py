import os
import time

from functools import reduce

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional

from joblib import Parallel
from joblib import delayed
from torch.nn.parameter import Parameter

from deepstruct.graph_class import GraphClass


from_position = {}
previous_positions = {}
node_id_for_new_layer = 1
member_weight = {}


def convolution_internal(
    value,
    value_temp,
    layer_id,
    kernel_size,
    stride,
    in_channels,
    input_dimension,
    output_dimension,
    input_dimension_sqr,
    output_dimension_sqr,
    output_single_channle_size,
    previous_positions,
    graph_append,
    to_append,
):

    to_node = (value * in_channels * kernel_size * kernel_size) + value
    self_node_id = to_node + 1
    to_append((layer_id, value, to_node))
    del to_append
    value_temp = value % output_dimension_sqr

    row_number = int(((value_temp * stride) + (kernel_size - 1)) / input_dimension)
    column_number = (
        int(
            ((((value % output_dimension) - 1) * stride) + kernel_size)
            % input_dimension
        )
        - 1
    )
    kernel_sqr = kernel_size * kernel_size
    out_chanel_index = int(value / output_single_channle_size) * kernel_sqr

    for input_kernel_number in range(in_channels):

        channel_row_number = row_number * stride
        in_chanel_index = input_kernel_number * kernel_sqr

        # if value % output_dimension_sqr == 0:
        #     value_temp = 0
        #     channel_row_number = 0
        from_position_value_0 = column_number + (
            input_kernel_number * (input_dimension_sqr)
        )
        for row_loop in range(kernel_size):
            from_position_value_1 = from_position_value_0 + (
                channel_row_number * input_dimension
            )
            for column_loop in range(kernel_size):

                weight_index = (
                    out_chanel_index
                    + in_chanel_index
                    + (row_loop * kernel_size)
                    + column_loop
                )
                weight_value = member_weight[weight_index]

                if (
                    weight_value <= 0
                ):  # Weights less hen threshold are already set zero.
                    continue

                from_position_value = from_position_value_1 + column_loop

                if from_position_value in previous_positions.keys():
                    old_position = previous_positions[from_position_value]
                    graph_append(
                        GraphClass(
                            old_position, layer_id, old_position, to_node, weight_value
                        )
                    )

                else:
                    if from_position_value in from_position:
                        graph_append(
                            GraphClass(
                                from_position[from_position_value],
                                layer_id,
                                from_position[from_position_value],
                                to_node,
                                weight_value,
                            )
                        )
                    else:
                        from_position[from_position_value] = self_node_id
                        graph_append(
                            GraphClass(
                                self_node_id,
                                layer_id,
                                self_node_id,
                                to_node,
                                weight_value,
                            )
                        )
                        self_node_id += 1

                # print(str(from_position_value) + " to Poistion: " + str(value))
            channel_row_number += 1

    del graph_append
    # value_temp += 1


def pooling_internal(
    value,
    layer_id,
    kernel_size,
    stride,
    input_dimension,
    output_dimension,
    graph_append,
    to_append,
):

    to_node = (value * kernel_size * kernel_size) + value
    self_node_id = to_node + 1

    to_append((layer_id, value, to_node))
    del to_append
    row_number = int(value / output_dimension)
    column_number = int(value % output_dimension)

    if column_number == 0:
        stride_for_increment = 0
    else:
        stride_for_increment = stride

    from_position_value_1 = (column_number * stride_for_increment) + (
        row_number * input_dimension * stride
    )
    global previous_positions

    for column_loop in range(kernel_size):
        for row_loop in range(kernel_size):

            from_position_value = (
                from_position_value_1 + row_loop + (column_loop * input_dimension)
            )

            if from_position_value in previous_positions.keys():
                old_position = previous_positions[from_position_value]
                graph_append(
                    GraphClass(old_position, layer_id, old_position, to_node, 0)
                )
            else:
                if from_position_value in from_position:
                    graph_append(
                        GraphClass(
                            from_position[from_position_value],
                            layer_id,
                            from_position[from_position_value],
                            to_node,
                            0,
                        )
                    )
                else:
                    from_position[from_position_value] = self_node_id
                    graph_append(
                        GraphClass(self_node_id, layer_id, self_node_id, to_node, 0)
                    )
                    self_node_id += 1
    del graph_append


# from tqdm import tqdm
# Wrapped tqdm for now as we might want to avoid another dependency
def tqdm(x):
    return x


class GraphGenerator:
    def __init__(self):
        self.__graph_structure = []
        self.__layer_id = 0
        self.__node_id = 1
        self.__to_position = []

    def generate_graph(self, local_trained_model, threshold, input_tensor):

        generated_graph = nx.DiGraph(model=local_trained_model.model_name)
        self.__graph_structure = []
        members = vars(local_trained_model).get("_modules")
        self.__layer_id = 0
        self.__node_id = 1
        self.__to_position = []
        add_edge_to_graph = generated_graph.add_edge
        global from_position

        for member in members.values():

            self.__layer_id += 1

            if isinstance(member, nn.Conv2d):
                input_tensor = self.__transform_conv2d_to_graph(
                    threshold, input_tensor, member
                )

            if isinstance(member, nn.MaxPool2d):
                input_tensor = self.__transform_maxpool_to_graph(
                    threshold, input_tensor, member, False
                )

            if isinstance(member, nn.AdaptiveAvgPool2d):
                input_tensor = self.__transform_maxpool_to_graph(
                    threshold, input_tensor, member, True
                )

            if isinstance(member, nn.Linear):
                input_tensor = self.__transform_linear_to_graph(
                    threshold, input_tensor, member
                )

            self.__to_position = [
                (lyr, c, n)
                for lyr, c, n in self.__to_position
                if lyr == self.__layer_id
            ]
            from_position = {}

        for graph in self.__graph_structure:
            add_edge_to_graph(
                graph.from_node,
                graph.to_node,
                weight=graph.weight,
                layer_id=graph.layer_id,
            )

        print("Graph generated successfully.")
        return generated_graph

    def __transform_conv2d_to_graph(self, threshold, input_tensor, member):

        in_channels = vars(member).get("in_channels")
        out_channels = vars(member).get("out_channels")
        kernel_size = vars(member).get("kernel_size")[0]
        stride = vars(member).get("stride")[0]
        padding = vars(member).get("padding")[0]
        dilation = vars(member).get("dilation")[0]

        graph_append = self.__graph_structure.append
        to_append = self.__to_position.append

        print(
            "IN_Channels = "
            + str(in_channels)
            + " Out_ channels = "
            + str(out_channels)
            + " Kernel_size = "
            + str(kernel_size)
            + " Stride: "
            + str(stride)
            + " Padding: "
            + str(padding)
            + " Dilation: "
            + str(dilation)
        )

        # Remove Unimportant weights
        zeros_output = torch.zeros(member.weight.shape)
        member.weight = Parameter(
            torch.where((member.weight > threshold), member.weight, zeros_output)
        )
        zeros_output = None

        global member_weight
        weight_size = reduce(lambda x, y: x * y, member.weight.size())
        member_weight = member.weight.view(weight_size, 1).squeeze().tolist()

        if padding > 0:
            input_tensor_padding = torch.nn.functional.pad(
                input_tensor, (padding, padding, padding, padding)
            )
            input_dimension = input_tensor_padding.size(3)
        else:
            input_dimension = input_tensor.size(3)

        input_tensor = member(input_tensor)  # Execution of convolution
        output_dimension = input_tensor.size(2)
        new_size_output = reduce(lambda x, y: x * y, input_tensor.size())

        global from_position
        from_position.clear()
        output_single_channle_size = new_size_output / out_channels

        global previous_positions
        temp_previous = [
            (c, n) for lyr, c, n in self.__to_position if lyr == (self.__layer_id - 1)
        ]
        previous_positions = dict(temp_previous)
        del temp_previous

        output_dimension_sqr = output_dimension * output_dimension
        input_dimension_sqr = input_dimension * input_dimension

        print(time.ctime())
        print(os.cpu_count())
        Parallel(n_jobs=os.cpu_count(), prefer="threads", require="sharedmem")(
            delayed(convolution_internal)(
                i,
                i,
                self.__layer_id,
                kernel_size,
                stride,
                in_channels,
                input_dimension,
                output_dimension,
                input_dimension_sqr,
                output_dimension_sqr,
                output_single_channle_size,
                previous_positions,
                graph_append,
                to_append,
            )
            for i in tqdm(range(new_size_output))
        )
        previous_positions.clear()
        print(time.ctime())
        print("Convolution layer Completed.")
        # for graph in self.__graph_structure:
        #     print("From Node: " + str(graph.from_node) + " To node; " + str(graph.to_node))
        return input_tensor

    def __transform_maxpool_to_graph(
        self, threshold, input_tensor, member, is_adaptive
    ):

        if is_adaptive:
            kernel_size = 2  # Hard coded for AlexNet Adaptive AvgPool layer
            stride = 1
            print("Adaptive Pooling started.")
        else:
            kernel_size = vars(member).get("kernel_size")
            stride = vars(member).get("stride")
            print("Pooling started.")

        print("kernel_size: " + str(kernel_size) + " Stride: " + str(stride))

        graph_append = self.__graph_structure.append
        to_append = self.__to_position.append

        input_dimension = input_tensor.size(2)
        input_tensor = member(input_tensor)  # Execution of MaxPool
        output_dimension = input_tensor.size(2)

        # Flat the output image from 2d to 1d
        new_size_output = reduce(lambda x, y: x * y, input_tensor.size())

        global from_position
        from_position.clear()

        global previous_positions
        temp_previous = [
            (c, n) for lyr, c, n in self.__to_position if lyr == (self.__layer_id - 1)
        ]
        previous_positions = dict(temp_previous)
        del temp_previous

        print(time.ctime())
        print(os.cpu_count())
        Parallel(n_jobs=os.cpu_count(), prefer="threads", require="sharedmem")(
            delayed(pooling_internal)(
                i,
                self.__layer_id,
                kernel_size,
                stride,
                input_dimension,
                output_dimension,
                graph_append,
                to_append,
            )
            for i in tqdm(range(new_size_output))
        )

        print(time.ctime())
        previous_positions.clear()
        print("Pooling Completed.")
        return input_tensor

    def __transform_linear_to_graph(self, threshold, input_tensor, member):
        in_features = vars(member).get("in_features")
        out_features = vars(member).get("out_features")
        print(
            "in_features = " + str(in_features) + " out_features = " + str(out_features)
        )
        new_size = reduce(lambda x, y: x * y, input_tensor.size())
        input_tensor = member(input_tensor.view(-1, new_size))
        linear_weights = vars(member).get("_parameters").get("weight")

        graph_append = self.__graph_structure.append
        to_append = self.__to_position.append

        # Remove Unimportant weights
        zeros_weight = torch.zeros(out_features, in_features)
        new_weights = torch.where(
            linear_weights > threshold, linear_weights, zeros_weight
        )

        from_position = {}
        to_counter = 0

        global previous_positions
        temp_previous = [
            (c, n) for lyr, c, n in self.__to_position if lyr == (self.__layer_id - 1)
        ]
        previous_positions = dict(temp_previous)
        del temp_previous

        for new_weight_1 in tqdm(new_weights):  # output layers node time run
            to_node = self.__node_id + 1
            self.__node_id += 1
            to_counter += 1
            counter = 0
            to_append((self.__layer_id, to_counter, to_node))
            for new_weight_2 in new_weight_1:  # input layers node time run
                counter += 1
                weight_value = new_weight_2.item()
                if weight_value != 0:

                    if (counter - 1) in previous_positions.keys():
                        old_position = previous_positions[(counter - 1)]
                        graph_append(
                            GraphClass(
                                old_position,
                                self.__layer_id,
                                old_position,
                                to_node,
                                weight_value,
                            )
                        )
                    else:
                        if (
                            counter in from_position
                        ):  # Prevent creating new node id for already existing node
                            graph_append(
                                GraphClass(
                                    from_position[counter],
                                    self.__layer_id,
                                    from_position[counter],
                                    to_node,
                                    weight_value,
                                )
                            )
                        else:
                            self.__node_id += 1
                            graph_append(
                                GraphClass(
                                    self.__node_id,
                                    self.__layer_id,
                                    self.__node_id,
                                    to_node,
                                    weight_value,
                                )
                            )
                            from_position[counter] = self.__node_id
        print("Linear layer completed.")
        return input_tensor
