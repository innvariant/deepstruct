import torch

from deepstruct.constants import DEFAULT_OPERATIONS
from deepstruct.node_map_strategies import NodeMapStrategy, HighLevelNodeMapListStrategy
from deepstruct.traverse_strategies import TraversalStrategy, FrameworkTraversal


class Transformer:
    def __init__(self, random_input, traversal_strategy: TraversalStrategy = None,
                 node_map_strategy: NodeMapStrategy = None, namespaces_relevant_ops=None):
        self.random_input = random_input
        if namespaces_relevant_ops is None:
            self.namespaces_relevant_ops = DEFAULT_OPERATIONS
        else:
            self.namespaces_relevant_ops = namespaces_relevant_ops
        if traversal_strategy is None:
            self.traversal_strategy = FrameworkTraversal()
        else:
            self.traversal_strategy = traversal_strategy
        if node_map_strategy is None:
            self.node_map_strategy = HighLevelNodeMapListStrategy()
        else:
            self.node_map_strategy = node_map_strategy

    def transform(self, model: torch.nn.Module):
        try:
            self.traversal_strategy.init(self.random_input, model, self.namespaces_relevant_ops, self.node_map_strategy)
            self.traversal_strategy.prepare_tensor_traversal()
            self.traversal_strategy.prepare_function_traversal()
            self.traversal_strategy.traverse()
        finally:
            self.traversal_strategy.restore_traversal()

    def get_graph(self):
        return self.traversal_strategy.get_graph()

