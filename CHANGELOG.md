# Changelog for deepstruct

## 0.8
* deprecation of learning utilities
* integrated additional normalization layers

## 0.7
* new minimal version requirement is python 3.7
* introduced interface for "functors" which transform a nn.Module into a directed acyclic graph
* created a first functor for Linear and MaskedLinear layers
* a graph transform class passes a random input through a generic module and can transform it into a graph given that it consists of linear or conv2d layers (first tests added)
* added mkdocs to provide an initial documentation skeleton

## 0.6
* introduced *BaseRecurrentLayer*, *MaskedRecurrentLayer*, *MaskedGRULayer*, *MaskedLSTMLayer*
* introduced *deepstruct.recurrent.MaskedDeepRNN* for sparse recurrent models

## 0.5
* new feature: concept of scalable families which is a first notion of *graph themes* analysis
* file restructuring for better semantics
* pypaddle will be renamed to deepstruct

## 0.4
* switched to poetry for dependency and build management
* added integration tests
* switched to pytest instead of unittest

## 0.3
* added support to define input shape for MaskedDeepFFN and MaskedDeepDAN
* changed parameter for recompute_mask(epsilon) to recompute_mask(theta) as it should denote a threshold
* implemented a first running version of a randomly wired cell network, more general than RandWireNN and in spirit of analysing graph theoretic properties
* bugfixes on generating structures from masks
* added/modified data loader utilities for mnist/cifar (probably no official part and concern of this library tools)
* fixed PyPi setup and tested installation routine
* defined networkx and torch as dependencies in setup.py. Next will be to check if it can be shadowed by pytorch packages from conda channels
* added a DeepCellDAN() which builds directed, acyclic networks with customized cells given a certain structure

## 0.2
* introduced LayeredGraph as a wrapper for directed graphs which provides access to its layered ordering
* central provided modules are MaskedLinearLayer, MaskedDeepFFN and MaskedDeepDAN
* provided first functionality to generate structures from masked modules
