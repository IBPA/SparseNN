# Overview #
a SparseBlock module takes SparseBlock table input (instead of tensor). Let us assume sensors S1, S2, ..., Sn,  each producing ...

There are two types of SparseBlock modules:
1. Both input and output are SparseBlock tables (e.g. SparseBlockReLU, SparseBlockLinear).
2. Input is SparseBlock table while output is tensor (e.g SparseBlockToDenseLinear, SparseBlockToDenseAdd)
