# deepstruct - tools for neural network graph topology analysis ![Tests](https://github.com/innvariant/deepstruct/workflows/Tests/badge.svg)  [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
Tools for fusing machine learning and graph theory.
We are fascinated with the interplay of end-to-end learnable, locally restricted models and their graph theoretical properties.
Searching for evidence of the structural prior hypothesis.
Interested in pruning, neural architecture search or learning theory in general?

**Note** we are currently renaming *deepstruct* to *deepstruct* due to naming collisions and better semantical wording.

## Installation
Via *poetry* (**recommended** for projects) using PyPi:
``poetry add deepstruct``

Directly with *pip* from PyPi:
```bash
pip install deepstruct
```

Via *conda* in your *environment.yml* (recommended for reproducible experiments):
```yaml
name: exp01
channels:
- defaults
dependencies:
- pip>=20
- pip:
    - deepstruct
```

From public GitHub:
```bash
pip install --upgrade git+ssh://git@github.com:innvariant/deepstruct.git
```

## Models
- *deepstruct.sparse.MaskableModule*: pytorch modules that contain explicit masks to enforce (mostly zero-ordered) structure
- *deepstruct.sparse.MaskedLinearLayer*: pytorch module with a simple linear layer extended with masking capability.
Suitable if you want to have linear-layers on which to enforce masks which could be obtained through pruning, regularization or other other search techniques.
- *deepstruct.sparse.MaskedDeepFFN*: feed-forward neural network with any width and depth and easy-to-use masks.
Suitable for simple and canonical pruning research on zero-ordered structure
- *deepstruct.sparse.MaskedDeepDAN*: feed-forward neural network with skip-layer connections based on any directed acyclic network.
Suitable for arbitrary structures on zero-order and on that level most flexible but also computationally expensive.
- *deepstruct.sparse.DeepCellDAN*: complex module based on a directed acyclic network and custom cells on third-order structures.
Suitable for large-scale neural architecture search

## What is the orders of structure?
- zero-th order: weight-level
- first order: kernel-level (filter, channel, blocks, cells)
- second order: layers

There is various evidence across empirical machine learning studies that the way artificial neural networks are structurally connected has a (minor?) influence on performance metrics such as the accuracy or probably even on more complex concepts such as adversarial robustness.
What do we mean by "structure"?
We define structure over graph theoretic properties given a computational graph with very restricted non-linearities.
This includes all major neural network definitions and lets us study them from the perspective of their *representation* and their *structure*.
In a probabilistic sense, one can interprete structure as a prior to the model and despite single-layered wide networks are universal function approximators we follow the hypothesis that given certain structural priors we can find models with better properties.



## Sparse Neural Network implementations
![Sparse Network Connectivity on zeroth order with a masked deep feed-forward neural network](doc/masked-deep-ffn.png)
![Sparse Network Connectivity on zeroth order with a masked deep neural network with skip-layer connections](doc/masked-deep-dan.png)
![Sparse Network Connectivity on second order with a masked deep cell-based neural network](doc/masked-deep-cell-dan.png)

**What's contained in deepstruct?**
- ready-to-use models in pytorch for learning instances on common (supervised/unsupervised) datasets from which a structural analysis is possible
- model-to-graph transformations for studying models from a graph-theoretic perspective

Before considering implementations, one should have a look on possible representations of Sparse Neural Networks.
In case of feed-forward neural networks (FFNs) the network can be represented as a list of weight matrices.
Each weight matrix represents the connections from one layer to the next.
Having a network without some connections then simply means setting entries in those matrices to zero.
Removing a particular neuron means setting all entries representing its incoming connections to zero.

However, sparsity can be employed on various levels of a general artificial neural network.
Zero order sparsity would remove single weights (representing connections) from the network.
First order sparsity removes groups of weights within one dimension of a matrix from the network.
Sparsity can be employed on connection-, weight-, block-, channel-, cell-level and so on.
Implementations respecting the areas for sparsification can have drastical differences.
Thus there are various ways for implementing Sparse Neural Networks.

### Feed-forward Neural Network with sparsity
The simplest implementation is probably one which provides multiple layers with binary masks for each weight matrix.
It doesn't consider any skip-layer connections.
Each layer is then connected to only the following one.
```python
import deepstruct.sparse

mnist_model = deepstruct.sparse.MaskedDeepFFN((1, 28, 28), 10, [100, 100])
```


```python
import deepstruct.sparse

structure  = deepstruct.sparse.CachedLayeredGraph()
# .. add nodes & edges to the networkx graph structure

# Build a neural network classifier with 784 input and 10 output neurons and the given structure
model = deepstruct.sparse.MaskedDeepDAN(784, 10, structure)
model.apply_mask()  # Apply the mask on the weights (hard, not undoable)
model.recompute_mask()  # Use weight magnitude to recompute the mask from the network
pruned_structure = model.generate_structure()  # Get the structure -- a networkx graph -- based on the current mask

new_model = deepstruct.sparse.MaskedDeepDAN(784, 10, pruned_structure)
```
```python
import deepstruct.sparse

model = deepstruct.sparse.MaskedDeepFFN(784, 10, [100, 100])
# .. train model
model.generate_structure()  # a networkx graph
```
