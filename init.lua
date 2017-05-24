require 'torch'
require 'nn'

sparsenn = {}

require('sparsenn.SparseBlockDropout')
require('sparsenn.SparseBlockFlattenDim3')
require('sparsenn.SparseBlockLinear')
require('sparsenn.SparseBlockReLU')
require('sparsenn.SparseBlockSum')
require('sparsenn.SparseBlockTemporalConvolution')
require('sparsenn.SparseBlockTemporalMaxPooling')
require('sparsenn.SparseBlockToDenseLinear')
require('sparsenn.SparseBlockToDenseMul')
require('sparsenn.SparseBlockToDenseSum')

return sparsenn
