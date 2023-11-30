from abc import abstractmethod, ABC
from functools import wraps

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from deepstruct.node_map_strategies import CustomNodeMap
from deepstruct.topologie_representation import LayeredGraph
from deepstruct.constants import DEFAULT_OPERATIONS

import torch
import torch.fx


class TraversalStrategy(ABC):

    @abstractmethod
    def init(self,
             node_map_strategy: CustomNodeMap,
             include_fn=None,
             exclude_fn=None,
             namespaces_with_functions: list[tuple[any, str]] = None):
        pass

    @abstractmethod
    def traverse(self,
                 input_tensor: torch.Tensor,
                 model: torch.nn.Module):
        pass

    @abstractmethod
    def restore_traversal(self):
        pass

    @abstractmethod
    def get_graph(self):
        pass


class FXTraversal(TraversalStrategy):

    def __init__(self):
        self.exclude_fn = None
        self.include_fn = None
        self.node_map_strategy = None
        self.layered_graph = None

    def init(self, node_map_strategy: CustomNodeMap, include_fn=None,
             exclude_fn=None, namespaces_with_functions: list[tuple[any, str]] = None):
        self.layered_graph = LayeredGraph()
        self.node_map_strategy = node_map_strategy
        self.include_fn = include_fn
        self.exclude_fn = exclude_fn

    def traverse(self, input_tensor: torch.Tensor, model: torch.nn.Module):
        traced = fx.symbolic_trace(model)
        traced_modules = dict(traced.named_modules())
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(traced).propagate(input_tensor)

        class EmptyShape:
            def __init__(self):
                self.shape = None

        for node in traced.graph.nodes:
            module_instance = traced_modules.get(node.target)
            shape = node.meta.get('tensor_meta', EmptyShape()).shape
            self.node_map_strategy.map_node(type(module_instance), self.layered_graph, shape, node.name)

    def restore_traversal(self):
        pass

    def get_graph(self):
        return self.layered_graph.graph


class FrameworkTraversal(TraversalStrategy):

    def __init__(self):
        self.functions_to_decorate = []
        self.layered_graph = LayeredGraph()
        self.orig_func_defs = []
        self.node_map_strategy: CustomNodeMap = None
        self.namespaces_with_functions: list[tuple[any, str]] = []

    def init(self, node_map_strategy: CustomNodeMap, include_fn=None,
             exclude_fn=None, namespaces_with_functions: list[tuple[any, str]] = None):
        self.node_map_strategy = node_map_strategy
        self.namespaces_with_functions = namespaces_with_functions if namespaces_with_functions else DEFAULT_OPERATIONS

    def traverse(self,
                 input_tensor: torch.Tensor,
                 model: torch.nn.Module, ):
        self._prepare_for_traversal()
        model.forward(input_tensor)

    def restore_traversal(self):
        for func_def in self.orig_func_defs:
            setattr(func_def[0], func_def[1], func_def[2])

    def get_graph(self):
        return self.layered_graph.graph

    def _prepare_for_traversal(self):
        for namespace, func_name in self.namespaces_with_functions:
            if not hasattr(namespace, func_name):
                print("Function: ", func_name, " not found in namespace: ", namespace)
            else:
                self.functions_to_decorate.append((namespace, func_name, getattr(namespace, func_name)))
                self.orig_func_defs.append((namespace, func_name, getattr(namespace, func_name)))

        for fn in self.functions_to_decorate:
            decorated_fn = self._decorate_functions(fn[2], fn[0], fn[1])
            setattr(fn[0], fn[1], decorated_fn)

    def _decorate_functions(self, func, func_namespace, func_name):
        # Problem: what if there is recurrence?
        @wraps(func)
        def decorator_func(*args, **kwargs):
            all_args = list(args) + list(kwargs.values())
            executed = False
            out = None
            for arg in all_args:
                if issubclass(type(arg), torch.Tensor):
                    input_shape = arg.shape
                    out = func(*args, **kwargs)
                    fn_name = func_namespace.__name__ + " " + func_name
                    shape = getattr(out, 'shape', None)
                    self.node_map_strategy.map_node(func_namespace, self.layered_graph, shape,
                                                    fn_name, input_shape)
                    executed = True
                    break  # what if there is another tensor e.g. add(t1, t2)
                    # Maybe show both?
            if not executed:
                print("No Tensor arguments where found in function: ", func_name, " namespace: ", func_namespace)
                out = func(*args, **kwargs)

            return out

        return decorator_func
