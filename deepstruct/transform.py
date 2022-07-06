import itertools
import warnings

from functools import partial

import numpy as np
import torch

from deepstruct.graph import LabeledDAG
from deepstruct.graph import LayeredGraph
from deepstruct.sparse import MaskedLinearLayer


def transform_mask_into_graph(graph: LayeredGraph, mask: torch.Tensor):
    assert mask.dtype == torch.bool
    assert graph is not None


class ForgetfulFunctor:
    def transform(self, model: torch.nn.Module) -> LabeledDAG:
        raise NotImplementedError("Abstract method needs to be implemented")

    def applies(self, model: torch.nn.Module):
        return True


class LinearLayerFunctor(ForgetfulFunctor):
    def __init__(self, threshold: float = None):
        self._threshold = threshold

    def transform_masked(self, model: MaskedLinearLayer):
        if self._threshold is not None:
            model.recompute_mask()

        return self.transform_mask(model.mask)

    def transform_linear(self, model: torch.nn.Linear):
        assert (
            self._threshold is not None
        ), "For transforming a linear layer you need to specify which threshold to use for pruning edges."

        in_features = model.in_features
        out_features = model.out_features
        mask = torch.ones((out_features, in_features), dtype=torch.bool)
        # TODO maybe also allow for non-L1-pruning methods?
        mask[torch.where(abs(model.weight) < self._threshold)] = False

        return self.transform_mask(mask)

    def transform_mask(self, mask: torch.tensor):
        assert mask is not None
        assert mask.dtype == torch.bool
        assert len(mask.shape) == 2

        dim_input = mask.shape[1]
        dim_output = mask.shape[0]

        graph = LabeledDAG()

        sources = graph.add_vertices(dim_input, layer=0)
        targets = graph.add_vertices(dim_output, layer=1)
        graph.add_edges_from(
            [
                (sources[s], targets[t])
                for (s, t) in itertools.product(
                    np.arange(dim_input), np.arange(dim_output)
                )
                if mask[t, s]
            ]
        )

        return graph

    def transform(self, model: torch.nn.Module):
        return (
            self.transform_masked(model)
            if isinstance(model, MaskedLinearLayer)
            else self.transform_linear(model)
        )

    def applies(self, model: torch.nn.Module):
        return isinstance(model, torch.nn.Linear) or isinstance(
            model, MaskedLinearLayer
        )


class Conv2dLayerFunctor(ForgetfulFunctor):
    def __init__(
        self, input_width: int = 0, input_height: int = 0, threshold: float = None
    ):
        self._input_width = input_width
        self._input_height = input_height
        self._threshold = threshold

    @property
    def width(self):
        return self._input_width

    @property
    def height(self):
        return self._input_height

    @width.setter
    def width(self, width: int):
        assert width > 0
        self._input_width = width

    @height.setter
    def height(self, height: int):
        assert height > 0
        self._input_height = height

    def transform(self, model: torch.nn.Module) -> LabeledDAG:
        # TODO does not respect sparsity in kernel currently
        assert isinstance(model, torch.nn.Conv2d)
        assert model.dilation == (
            1,
            1,
        ), "Currently dilation is not considered in this implementation"
        assert (
            self.width > 0 and self.height > 0
        ), "You need to specify input width and height for this functor."

        channels_in = model.in_channels
        channels_out = model.out_channels
        size_kernel = model.kernel_size
        stride = model.stride
        padding = model.padding

        assert (
            stride[0] > 0 and stride[1] > 0
        ), "Stride must be a natural number and at least be one"

        graph = LabeledDAG()
        input_neurons = graph.add_vertices(
            channels_in * self._input_width * self._input_height, layer=0
        )

        def output_shape(size, dim):
            return int(
                np.floor((size - size_kernel[dim] + 2 * padding[dim]) / stride[dim]) + 1
            )

        output_height = output_shape(self._input_height, 0)
        output_width = output_shape(self._input_width, 1)
        # print("output height", output_height)
        # print("output width", output_width)
        output_neurons = graph.add_vertices(
            channels_out * output_height * output_width,
            layer=1,
        )

        # print(len(input_neurons))
        # print(len(output_neurons))

        def get_input_neuron(channel: int, row: int, col: int):
            return int(
                input_neurons[
                    int(
                        (col * self._input_height + row)
                        + (channel * self._input_width * self._input_height)
                    )
                ]
            )

        def get_output_neuron(channel_out: int, row: int, col: int):
            return int(
                output_neurons[
                    int(
                        (col * output_height + row)
                        + (channel_out * output_width * output_height)
                    )
                ]
            )

        # print("input width", self._input_width)
        # print("input height", self._input_height)
        # print("stride", stride)
        # print("kernel", size_kernel)

        for idx_channel_out in range(channels_out):
            # print()
            # print("Channel", idx_channel_out)
            for idx_channel_in in range(channels_in):
                out_col = 0
                offset_height = -padding[0]
                # print("Offset height initial:", offset_height)
                while offset_height + size_kernel[0] <= self._input_height:
                    # print("Offset height", offset_height)
                    out_row = 0
                    offset_width = -padding[1]
                    # print("Offset width initial:", offset_width)
                    while offset_width + size_kernel[1] <= self._input_width:
                        # print("Offset width", offset_width)
                        # print("out:", (out_row, out_col), end="")
                        target = get_output_neuron(idx_channel_out, out_row, out_col)
                        # print("", target)
                        # tmp_ll = []
                        # print(list(range(max(0, offset_height), min(offset_height+size_kernel[0], self._input_height))))
                        # print(list(range(max(0, offset_width), min(offset_width+size_kernel[1], self._input_width))))
                        # print("colrange(", max(0, offset_height), min(offset_height + size_kernel[0], self._input_height), ")")
                        edges = []
                        for col in range(
                            max(0, offset_height),
                            min(offset_height + size_kernel[0], self._input_height),
                        ):
                            # print("rowrange(", max(0, offset_width), min(offset_width + size_kernel[1], self._input_width), ")")
                            for row in range(
                                max(0, offset_width),
                                min(offset_width + size_kernel[1], self._input_width),
                            ):
                                source = get_input_neuron(idx_channel_in, row, col)
                                edges.append((source, target))
                                # tmp_ll.append((source, target))
                        graph.add_edges_from(edges)
                        # print(tmp_ll)
                        # if len(tmp_ll) != size_kernel[0]*size_kernel[1]:
                        # print("---------------- len(tmp_ll) !=", size_kernel[0]*size_kernel[1])
                        offset_width += stride[1]
                        out_row += 1
                    # print("o1: ", offset_width)
                    # print("o2: ", self._input_width + padding[1])
                    offset_height += stride[0]
                    out_col += 1
                # print()

        return graph

    def applies(self, model: torch.nn.Module):
        return isinstance(model, torch.nn.Conv2d)


def forward_capture_shape(obj, orig_forward, input, **kwargs):
    # print("forward_capture_shape()")
    # print("input", input)
    # print(input.shape)
    obj._deepstruct_input_shape = input.shape
    return orig_forward(input)


class GraphTransform(ForgetfulFunctor):
    """
    Standard zeroth-order transformation from neural networks to graphs.
    """

    def __init__(self, random_input: torch.Tensor):
        self.random_input = random_input
        self._pointwise_ops = [
            torch.nn.Threshold,
            torch.nn.ReLU,
            torch.nn.RReLU,
            torch.nn.Hardtanh,
            torch.nn.ReLU6,
            torch.nn.Sigmoid,
            torch.nn.Tanh,
            torch.nn.ELU,
            torch.nn.CELU,
            torch.nn.SELU,
            torch.nn.GLU,
            torch.nn.GELU,
            torch.nn.Hardshrink,
            torch.nn.LeakyReLU,
            torch.nn.LogSigmoid,
            torch.nn.Softplus,
            torch.nn.Softshrink,
            torch.nn.MultiheadAttention,
            torch.nn.PReLU,
            torch.nn.Softsign,
            torch.nn.Tanhshrink,
            torch.nn.Softmin,
            torch.nn.Softmax,
            torch.nn.Softmax2d,
            torch.nn.LogSoftmax,
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]

    @property
    def random_input(self):
        return self._random_input

    @random_input.setter
    def random_input(self, random_input: torch.Tensor):
        assert random_input is not None
        assert hasattr(random_input, "shape")
        self._random_input = random_input

    def _punch(self, module: torch.nn.Module):
        for child in module.children():
            self._punch(child)
        setattr(
            module, "forward", partial(forward_capture_shape, module, module.forward)
        )
        return module

    def _transform_partial(self, module: torch.nn.Module, graph: LabeledDAG):
        assert graph is not None
        functor_conv = Conv2dLayerFunctor()
        functor_linear = LinearLayerFunctor(threshold=0.01)

        partial = None
        if isinstance(module, torch.nn.ModuleList) or isinstance(
            module, torch.nn.Sequential
        ):
            # partial = self.transform(module)
            # graph.append(partial)
            for child in module:
                graph = self._transform_partial(child, graph)
        elif isinstance(module, torch.nn.Linear):
            partial = functor_linear.transform(module)
            graph.append(partial)
        elif isinstance(module, torch.nn.Conv2d):
            width = module._deepstruct_input_shape[-1]
            height = module._deepstruct_input_shape[-2]
            functor_conv.width = width
            functor_conv.height = height
            partial = functor_conv.transform(module)
            graph.append(partial)
        elif isinstance(module, torch.nn.Dropout):
            # Dropout behaves structurally like a linear-layer and we ignore the fact for now that some edges
            # are ignored probabilistically
            pass
        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            # TODO pooling needs to be transformed; most pooling results in structural singularities
            pass
        elif any(isinstance(module, op) for op in self._pointwise_ops):
            # Point-wise operations (mostly activation functions) do not change the structure
            # except for applying non-linear transformations on the input
            pass
        else:
            warnings.warn(f"Warning: ignoring sub-module of type {type(module)}")

        return graph

    def transform(self, model: torch.nn.Module):
        graph = LabeledDAG()

        self._punch(model)
        model.forward(self.random_input)
        graph.add_vertices(np.prod(self.random_input.shape), layer=0)

        for module in model.children():
            graph = self._transform_partial(module, graph)

        return graph

    def applies(self, model: torch.nn.Module):
        return all(
            isinstance(c, torch.nn.Linear) or isinstance(c, torch.nn.Conv2d)
            for c in model.children()
        )
