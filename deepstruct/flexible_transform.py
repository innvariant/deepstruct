import networkx
import torch

from deepstruct.node_map_strategies import CustomNodeMap, HighLevelNodeMap
from deepstruct.traverse_strategies import TraversalStrategy, FXTraversal


class GraphTransform:
    def __init__(self, random_input, traversal_strategy: TraversalStrategy = FXTraversal(),
                 node_map_strategy: CustomNodeMap = HighLevelNodeMap()):
        self.random_input = random_input
        self.traversal_strategy = traversal_strategy
        self.node_map_strategy = node_map_strategy

    def transform(self, model: torch.nn.Module):
        try:
            self.traversal_strategy.init(self.node_map_strategy)
            self.traversal_strategy.traverse(self.random_input, model)
        finally:
            self.traversal_strategy.restore_traversal()

    def get_graph(self) -> networkx.DiGraph:
        return self.traversal_strategy.get_graph()
