import itertools

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

        return self.transform_mask(model.get_mask())

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

    def transform(self, model: torch.nn):
        return (
            self.transform_masked(model)
            if isinstance(model, MaskedLinearLayer)
            else self.transform_linear(model)
        )

    def applies(self, model: torch.nn.Module):
        return isinstance(model, torch.nn.Linear) or isinstance(
            model, MaskedLinearLayer
        )


class GraphTransform(ForgetfulFunctor):
    """
    Standard zeroth-order transformation from neural networks to graphs.
    """

    def transform(self, model: torch.nn):
        pass
