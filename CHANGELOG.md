# Changelog for pypaddle

# Development Version

## 0.3
* added support to define input shape for MaskedDeepFFN and MaskedDeepDAN
* changed parameter for recompute_mask(epsilon) to recompute_mask(theta) as it should denote a threshold
* implemented a first running version of a randomly wired cell network, more general than RandWireNN and in spirit of analysing graph theoretic properties
* bugfixes on generating structures from masks
* added/modified data loader utilities for mnist/cifar (probably no official part and concern of this library tools)
* fixed PyPi setup and tested installation routine

## 0.2
* introduced LayeredGraph as a wrapper for directed graphs which provides access to its layered ordering
* central provided modules are MaskedLinearLayer, MaskedDeepFFN and MaskedDeepDAN
* provided first functionality to generate structures from masked modules
