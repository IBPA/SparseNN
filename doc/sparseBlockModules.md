# Overview #
Assuming we have sensors S1, S2, ... , Sn and a sensor Y. The size of the signal from different sensors maybe different while all signals from a particular sensor  have the same size. In a given event, the Y sensor **will** have a signal while a given Si sensor **may** have a signal. Given such sensory data (collected from various events), number of learning tasks can be defined (we mention two here):
A. Train a neural network to predict the signal from Y sensor using available information from Si sensors.
B. Given "A." rank the sensors Si, with respect to the information each provide about sensor Y (in the context of the observed events).

A SparseBlock module takes SparseBlock table input (instead of tensor).
There are two types of SparseBlock modules:
1. Both input and output are SparseBlock tables (e.g. SparseBlockReLU, SparseBlockLinear).
2. Input is SparseBlock table while output is tensor (e.g SparseBlockToDenseLinear, SparseBlockToDenseAdd)
