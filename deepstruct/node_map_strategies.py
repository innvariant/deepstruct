from abc import abstractmethod
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

    def __init__(self, threshold=0.01):
        super().__init__({
            torch.nn.Linear: Linear2LayerMapper(threshold),
        }, All2VertexNodeMapper())


class All2VertexNodeMapper(NodeMapper):

    def add_node(self, graph, predecessors, **kwargs):
        node_name = kwargs.pop('name')
        graph.add_vertex(node_name, **kwargs)
        graph.add_edges(predecessors, node_name)


class Linear2LayerMapper(NodeMapper):

    def __init__(self, threshold):
        self.name_index = {}
        self.threshold = threshold

    def add_node(self, graph, predecessors, **kwargs):
        model = kwargs.get("origin_module")
        in_features = model.in_features
        out_features = model.out_features
        mask = torch.ones((out_features, in_features), dtype=torch.bool)
        mask[torch.where(abs(model.weight) < self.threshold)] = False
        for i in range(in_features):
            kwargs['mask'] = mask[:, i]
            self._add_node(graph, predecessors, **kwargs)

    def _add_node(self, graph, predecessors, **kwargs):
        node_name = kwargs.pop('name')
        graph.add_vertex(node_name, **kwargs)
        graph.add_edges(predecessors, node_name)
