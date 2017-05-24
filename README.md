# SparseNN

This package provides neural network modules supporting sparse data in various levels using [Torch](https://github.com/torch/torch7/blob/master/README.md) (future support for sparse models would be desired)

### Motivation
When it comes to neural networks, the support for sparisity is often limitted to input layers only (e.g. [SparseLinear](https://github.com/torch/nn/blob/master/doc/simple.md#nn.SparseLinear)). This repository is intended for development of NN modules supporting sparsity in various layers (e.g. modules that preserve sparsity).

## To Install
* git clone https://github.com/ameenetemady/SparseNN/
* cd SparseNN
* luarocks make rocks/sparsenn-scm-1.rockspec
