# deepstruct
Documentation for [deepstruct on github](https://github.com/innvariant/deepstruct).

### Notes
- **2020-11-22** currently documenting the package


## Introduction


## A simple example
```python
import deepstruct.sparse

input_size = 5
output_size = 2
structure  = deepstruct.sparse.CachedLayeredGraph()
structure.add_nodes_from(range(20))
model = deepstruct.sparse.MaskedDeepDAN(input_size, output_size, structure)
```

## Available Models


## Artificial Datasets
