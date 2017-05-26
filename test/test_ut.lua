-- To run simple do: th -lsparsenn <this_file_name>

local testUtil = require('./testUtil.lua')
local mytest = torch.TestSuite()
local tester = torch.Tester()

local taSBInputA = { nBatchSize = 4,
                   taData = { { teRowIdx = torch.LongTensor({{2}, {4}}), -- nRow: 2, nCol: 7, nChanel:2
                                teValue = torch.Tensor({{{1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}, {1, 10}},
                                                        {{1, 10}, {1, 10}, {1, 10}, {2, 20}, {0, 0}, {0, 0}, {0, 0}}}) },

                               { teRowIdx = torch.LongTensor({{2}}), -- nRow: 1, nCol: 3, nChanel:2
                                 teValue = torch.Tensor({ {{0, 0}, {-1, -10}, {1, 10}} }) }
                    }
                  }

function mytest.SparseBlockDropout() -- mimicing same test of nn.Dropout
   local p = 0.2
   local taInput = { nBatchSize = 4, 
                    taData = { { teRowIdx = torch.LongTensor({{2}, {4}}), 
                                 teValue = torch.Tensor(2, 500):fill(1-p) },

                               { teRowIdx = torch.LongTensor({{2}}),
                                 teValue = torch.Tensor(1, 1000):fill(1-p) }
                    }
                  }
   local mNet = nn.SparseBlockDropout(p)

   --forward
   local taOutput = mNet:forward(taInput)
   for k, taV in pairs(taOutput.taData) do
      tester:assert(math.abs(taV.teValue:mean() - (1-p)) < 0.05, 'dropout output')
   end

   --backward
   local taGradInput = mNet:backward(taInput, taInput)
   for k, taV in pairs(taOutput.taData) do
      tester:assert(math.abs(taV.teValue:mean() - (1-p)) < 0.05, 'dropout output')
   end
end

function mytest.SparseBlockTemporalConvolution()
   local taInput = testUtil.SparseBlockClone(taSBInputA)

   local mNet = nn.SparseBlockTemporalConvolution(2, 2, 3)
   mNet.weight[1] = mNet.weight[1]:fill(1)
   mNet.weight[2] = mNet.weight[2]:fill(2)

   -- forward 
   local taOutput = mNet:forward(taInput)
   local taMatch = { 891, 0 } -- the validated sum for each column
   for k, taV in pairs(taOutput.taData) do
      tester:asserteq(taMatch[k], taV.teValue:sum(), " forward error")
   end

   -- backward  (zero case)
   local taCopy = testUtil.SparseBlockClone(taOutput)
   testUtil.SparseBlockApply(taCopy, function(x) x:fill(0) end)
   local taGradInput = mNet:backward(taInput, taCopy)
   for k, taV in pairs(taGradInput.taData) do
      tester:asserteq(0, taV.teValue:nonzero():nElement(), " backward error")
   end
   tester:asserteq(0, mNet.gradWeight:nonzero():nElement(), " gradWeight error ")

   -- ToDo: test shared weight updates
end

function mytest.SparseBlockReLU()
   local taInput = testUtil.SparseBlockClone(taSBInputA)
   local mNet = nn.SparseBlockReLU()
   
   -- forward (defaults)
   local taOutput = mNet:forward(taInput)
   local taMatch = { 132, 11 } -- the validated sum for each column
   for k, taV in pairs(taOutput.taData) do
      tester:asserteq(taMatch[k], taV.teValue:sum(), " forward error")
   end

   -- backward
   local taGradOutput = testUtil.SparseBlockApply(testUtil.SparseBlockClone(taInput), 
                                                  function(x) 
                                                     x:fill(1) 
                                                  end)

   local taGradInput = mNet:backward(taInput, taGradOutput)
   local taMatch = { 22, 2 } -- the validated sum for each column
   for k, taV in pairs(taGradInput.taData) do
      tester:asserteq(taMatch[k], taV.teValue:sum(), " forward error")
   end
end

tester:add(mytest)
tester:run()
