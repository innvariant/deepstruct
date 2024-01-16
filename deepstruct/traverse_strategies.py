import math
import operator
from abc import abstractmethod, ABC
from types import ModuleType
from typing import Tuple, Dict, Optional, Any, Union, Callable

import numpy as np
import numpy.random
from torch.fx.node import Node


from deepstruct.node_map_strategies import CustomNodeMap
from deepstruct.topologie_representation import LayeredFXGraph

import torch
import torch.fx


class TraversalStrategy(ABC):

    @abstractmethod
    def init(self,
             node_map_strategy: CustomNodeMap):
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

    def __init__(self, distribution_fn=np.random.normal, include_fn=None, include_modules=None, exclude_fn=None,
                 exclude_modules=None, fold_modules=None, unfold_modules=None):
        self.traced_model = None
        self.distribution_fn = distribution_fn
        self.include_fn = include_fn
        self.include_modules = include_modules
        self.exclude_fn = exclude_fn
        self.exclude_modules = exclude_modules
        self.node_map_strategy = None
        self.layered_graph = LayeredFXGraph()
        self.fold_modules = fold_modules
        self.unfold_modules = unfold_modules

    def init(self, node_map_strategy: CustomNodeMap):
        self.node_map_strategy = node_map_strategy

    def traverse(self, input_tensor: torch.Tensor, model: torch.nn.Module):
        dist_fn = self.distribution_fn
        unfold = self.unfold_modules if self.unfold_modules else []
        fold = self.fold_modules if self.fold_modules else []

        class CustomTracer(torch.fx.Tracer):

            def __init__(self,
                         autowrap_modules: Tuple[ModuleType] = (math,),
                         autowrap_functions: Tuple[Callable, ...] = (),
                         param_shapes_constant: bool = False, ):
                super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)
                self.orig_mod = None

            def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, *_, **__):
                operators = [operator.gt, operator.ge, operator.lt, operator.le, operator.eq, operator.ne]
                if target and target in operators:
                    return dist_fn(0, 1) > 0.5
                return super().create_proxy(kind, target, args, kwargs, name, type_expr)

            def create_node(self, kind: str, target: Union[str, Callable],
                            args: Tuple[Any], kwargs: Dict[str, Any], name: Optional[str] = None,
                            type_expr: Optional[Any] = None) -> Node | None:
                n = super().create_node(kind, target, args, kwargs, name)
                if self.orig_mod is None:
                    n.orig_mod = target
                else:
                    n.orig_mod = self.orig_mod
                    self.orig_mod = None
                return n

            def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...],
                            kwargs: Dict[str, Any]) -> Any:
                self.orig_mod = m
                return super().call_module(m, forward, args, kwargs)

            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                if any(isinstance(m, fm) for fm in fold):
                    return True
                elif any(isinstance(m, um) for um in unfold):
                    return False
                else:
                    return super().is_leaf_module(m, module_qualified_name)

        traced_graph = CustomTracer().trace(model)
        traced = torch.fx.GraphModule(model, traced_graph)
        traced_modules = dict(traced.named_modules())
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(traced).propagate(input_tensor)
        self.traced_model = traced
        print(" ")  # testing purpose delete later
        traced.graph.print_tabular()  # testing purpose delete later

        class EmptyShape:
            def __init__(self):
                self.shape = None

        for node in traced.graph.nodes:
            if self._should_be_included(node):
                module_instance = traced_modules.get(node.target)
                shape = getattr(node.meta.get('tensor_meta', EmptyShape()), 'shape', None)
                self.node_map_strategy.map_node(
                    self.layered_graph,
                    type(module_instance),
                    node.args,
                    name=node.name,
                    shape=shape,
                    origin_module=node.orig_mod
                )
            else:
                self.layered_graph.ignored_nodes.append(node.name)

    def _should_be_included(self, node):
        if node.op == 'placeholder' or node.op == 'output':
            return True
        elif node.op == 'get_attr':
            return False
        else:
            return self._is_in_include(node) and not self._is_in_exclude(node)

    def _is_in_include(self, node):
        include_fn = self.include_fn if self.include_fn else []
        include_modules = self.include_modules if self.include_modules else []
        if len(include_fn) == 0 and len(include_modules) == 0:
            return True
        if node.op == 'call_module' and len(include_modules) > 0:
            return any(isinstance(node.orig_mod, m) for m in include_modules)
        elif len(include_fn) > 0:
            return any(node.orig_mod == f or node.orig_mod == getattr(f, '__name__', None) for f in include_fn)
        else:
            return True

    def _is_in_exclude(self, node):
        exclude_fn = self.exclude_fn if self.exclude_fn else []
        exclude_modules = self.exclude_modules if self.exclude_modules else []
        if len(exclude_modules) == 0 and len(exclude_fn) == 0:
            return False
        if node.op == 'call_module':
            return any(isinstance(node.orig_mod, m) for m in exclude_modules)
        else:
            node_name = getattr(node.orig_mod, '__name__', "name not found in orig")
            if node_name is "name not found in orig":
                node_name = str(node.orig_mod)
            return any(node.orig_mod == f or node_name
                       == getattr(f, '__name__', "name not found in fn") for f in exclude_fn)

    def restore_traversal(self):
        pass

    def get_graph(self):
        return self.layered_graph

