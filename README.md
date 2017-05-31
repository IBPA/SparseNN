# SparseNN #

This package provides neural network modules supporting sparse data in various levels using [Torch](https://github.com/torch/torch7/blob/master/README.md).


## Installation ##
* git clone https://github.com/ameenetemady/SparseNN/
* cd SparseNN
* luarocks make rocks/sparsenn-scm-1.rockspec

## Motivation ##
When it comes to neural networks, the support for sparsity is often limitted to input layers only (e.g. [SparseLinear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.SparseLinear)). This repository is intended for development of NN modules supporting sparsity in various layers (e.g. modules that preserve sparsity).

## Documentation ##
Depending on the problem structure a different "sparsity-aware" implementation maybe desired.

### SparseBlock ###
The first such implementation focused here is [SparseBlock](https://github.com/ameenetemady/SparseNN/blob/master/doc/sparseBlockModules.md). 
