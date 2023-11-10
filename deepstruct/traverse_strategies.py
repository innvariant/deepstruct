from abc import abstractmethod
from functools import wraps

import torch

from deepstruct.node_map_strategies import NodeMapStrategy
from deepstruct.topologie_representation import LayeredGraph


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