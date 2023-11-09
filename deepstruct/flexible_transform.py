from functools import wraps

import networkx
import torch
import networkx as nx
from torch import nn
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class LayeredGraph:

    def __init__(self):
        self.graph = networkx.DiGraph()
        self.vertex_id = 0
        self.layer_before = []

    def add_vertex(self, name):
        self.graph.add_node(self.vertex_id, name=name)
        if len(self.layer_before) > 0:
            for vertex in self.layer_before:
                self.graph.add_edge(vertex, self.vertex_id)
        self.layer_before = [self.vertex_id]
        self.vertex_id += 1

    def add_vertices(self, name, count):
        current_vertices = []
        for i in range(count):
            self.graph.add_node(self.vertex_id, name=name)
            current_vertices.append(self.vertex_id)
            self.vertex_id += 1
        if len(self.layer_before) > 0:
            for vertex in self.layer_before:
                for new_vertex in current_vertices:
                    self.graph.add_edge(vertex, new_vertex)
        self.layer_before = current_vertices

    def add_edge(self, src, dest):
        self.graph.add_edge(src, dest)


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


class TraversalStrategy:

    @abstractmethod
    def init(self,
             input_tensor: torch.Tensor,
             model: torch.nn.Module,
             namespaces_with_functions: list[tuple[any, str]],
             node_map_strategy: NodeMapStrategy):
        pass

    @abstractmethod
    def prepare_tensor_traversal(self):
        pass

    @abstractmethod
    def prepare_function_traversal(self):
        pass

    @abstractmethod
    def traverse(self):
        pass

    @abstractmethod
    def restore_traversal(self):
        pass

    @abstractmethod
    def get_graph(self):
        pass


class FrameworkTraversal(TraversalStrategy):

    def __init__(self):
        self.functions_to_decorate = []
        self.layered_graph = LayeredGraph()
        self.orig_func_defs = []
        self.model: torch.nn.Module = None
        self.input_tensor: torch.Tensor = None
        self.node_map_strategy: NodeMapStrategy = None
        self.namespaces_with_functions: list[tuple[any, str]] = []

    def init(self,
             input_tensor: torch.Tensor,
             model: torch.nn.Module,
             namespaces_with_functions: list[tuple[any, str]],
             node_map_strategy: NodeMapStrategy):
        self.input_tensor = input_tensor
        self.model = model
        self.namespaces_with_functions = namespaces_with_functions
        self.node_map_strategy = node_map_strategy

    def prepare_tensor_traversal(self):
        pass
        # input_tensor.tensor_deepstruct_graph = LayeredGraph()

    def prepare_function_traversal(self):
        for namespace, func_name in self.namespaces_with_functions:
            if not hasattr(namespace, func_name):
                print("Function: ", func_name, " not found in namespace: ", namespace)
            else:
                self.functions_to_decorate.append((namespace, func_name, getattr(namespace, func_name)))
                self.orig_func_defs.append((namespace, func_name, getattr(namespace, func_name)))

        for fn in self.functions_to_decorate:
            decorated_fn = self._decorate_functions(fn[2])
            setattr(fn[0], fn[1], decorated_fn)

    def traverse(self):
        self.model.forward(self.input_tensor)

    def restore_traversal(self):
        for func_def in self.orig_func_defs:
            setattr(func_def[0], func_def[1], func_def[2])

    def _decorate_functions(self, func):
        func_namespace = None
        func_name = None
        for func_def in self.functions_to_decorate:
            if func_def[2] == func:
                func_namespace = func_def[0]
                func_name = func_def[1]
                break

        @wraps(func)
        def decorator_func(*args, **kwargs):
            all_args = list(args) + list(kwargs.values())
            executed = False
            out = None
            for arg in all_args:
                if issubclass(type(arg), torch.Tensor):
                    input_shape = arg.shape
                    out = func(*args, **kwargs)
                    self.node_map_strategy.map_node(func_namespace, self.layered_graph, out.shape,
                                                    func_name, input_shape)
                    executed = True
                    break
            if not executed:
                print("No Tensor arguments where found in function: ", func_name, " namespace: ", func_namespace)
                out = func(*args, **kwargs)

            return out

        return decorator_func

    def get_graph(self):
        return self.layered_graph.graph


class Transformer:
    def __init__(self, random_input, traversal_strategy: TraversalStrategy = None,
                 node_map_strategy: NodeMapStrategy = None, namespaces_relevant_ops=None):
        self.random_input = random_input
        if namespaces_relevant_ops is None:
            self.namespaces_relevant_ops = [
                (torch, "add"),
                (torch.Tensor, "add"),
                (torch, "cos"),
                (torch.nn.modules.conv.Conv2d, "forward"),
                (torch.nn.modules.Linear, "forward"),
                (torch.nn.modules.MaxPool2d, "forward"),
                (torch.nn.modules.Flatten, "forward"),
                (torch.nn.modules.BatchNorm2d, "forward"),
                (torch.nn.functional, "relu")
            ]
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


if __name__ == '__main__':
    import torch.nn.functional as F


    def plot_graph(graph, title):
        labels = {node: graph.nodes[node]['name'] for node in graph.nodes()}
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(graph, labels=labels, with_labels=True, node_size=700, node_color='lightblue', font_size=8,
                ax=ax)
        plt.title(title)
        plt.show()


    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc = nn.Linear(54, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = torch.cos(x)
            return x


    net = SimpleCNN()
    input_tensor = torch.randn(1, 1, 6, 6)
    graph_transformer = Transformer(input_tensor)
    graph_transformer.transform(net)
    plot_graph(graph_transformer.get_graph(), "Transformation")
