# SparseBlock
This page describes the SparseBlock format and supported NN modules.
## Overview
Consider sensors S1, S2, ... , Sn and a sensor Y (as in figure bellow). Here the size of the signal from different sensors maybe different while all signals from a particular sensor have the same size. In a given event (corresponding to a row), the Y sensor **will** have a signal while a given Si sensor **may** have a signal. Given such sensory data (collected from various events), learning tasks can be defined:
* Train a NN to predict the Y sensor signal using Si sensor signals.
* Rank the sensors Si, with respect to the information each provide about sensor Y.

<img src="./SparseBlockData_A.png"  width="450">

For such tasks, using a non-sparse representation as in the above figure, will be inefficient. Hence a sparse representation called **SparseBlock** is used as shown in figure bellow (and described next).

<img src="./SparseBlockData_B.png"  width="450">

## Data Format
The data for Si sensors would be represented in SparseBlock format using lua tables and torch tensors as following:

```lua
{ nBatchSize = m -- Total number of events in the dataset
  taData = { {teRowIdx = teIdxSet1, -- A (nS1 x 1) LongTensor which holds the event ids for which S1 has signal 
              teValue = teSignalSet1 -- A (nS1 x nSizeS1) Tensor whih holds the corresponding signal values in teIdxSet1
             },
             {teRowIdx = teIdxSet2, -- A (nS2 x 1) LongTensor which holds the event ids for which S2 has signal 
              teValue = teSignalSet2 -- A (nS2 x nSizeS2) Tensor whih holds the corresponding signal values in teIdxSet2
             },
         --[[.
             .
             . --]]
             {teRowIdx = teIdxSetN, -- A (nSN x 1) LongTensor which holds the event ids for which Sn has signal 
              teValue = teSignalSetN -- A (nSN x nSizeSN) Tensor whih holds the corresponding signal values in teIdxSetN
             }
           }
}
```
For the Y sensor data however, a single torch tensor with m rows is used where m is the number of events.

## Modules ##
A SparseBlock module allways takes input in SparseBlock format (instead of tensor). When it comes to output format however, there are two types of modules:
1. *Symmetric:* output is in SparseBlock format hence only SparseBlock nn modules can be stacked on top of them. For example SparseBlockReLU would be considered symmetric.
2. *Asymmetric:* output is torch tensor tensor hence any torch nn module can be stacked on top of them. For example SparseBlockToDenseAdd would be considered asymmetirc.

For calculation of local gradients using backpropagaion, the [nn module](https://github.com/torch/nn/blob/master/doc/module.md) torch api is followed.


