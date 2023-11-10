from abc import abstractmethod


class NodeMapStrategy:

    @abstractmethod
    def map_node(self, func_module, graph, output_shape, func_name, input_shape=None):
        pass


class NodeMapper:

    @abstractmethod
    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class HighLevelNodeMapListStrategy(NodeMapStrategy):

    def __init__(self):
        self.high_level_mapper = All2VertexNodeMapper()

    def map_node(self, func_module, graph, output_shape, func_name, input_shape=None):
        node_name = func_module.__name__ + " " + func_name
        self.high_level_mapper.add_node(graph, output_shape, node_name, input_shape)


class CustomNodeMapListStrategy(NodeMapStrategy):
    """ The node_mapper_strategies must contain the full_namespace.class as key and a desired node-mapper
     strategy as value """

    def __init__(self, node_mapper_strategies: dict, default_strategy: NodeMapper):
        self.node_mapper_strategies = node_mapper_strategies
        self.default_strategy = default_strategy

    def map_node(self, func_module, graph, output_shape, func_name, input_shape=None):
        if func_module in self.node_mapper_strategies.keys():
            self.node_mapper_strategies.get(func_module).add_node(graph, output_shape, func_name, input_shape)
        else:
            self.default_strategy.add_node(graph, output_shape, func_name, input_shape)
        pass


class All2VertexNodeMapper(NodeMapper):

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        graph.add_vertex(func_name)


class Linear2LayerMapper(NodeMapper):
    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class Linear2VertexMapper(NodeMapper):
    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class Conv2LayerMapper(NodeMapper):

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass


class Conv2VertexMapper(NodeMapper):

    def add_node(self, graph, output_shape, func_name, input_shape=None):
        pass
