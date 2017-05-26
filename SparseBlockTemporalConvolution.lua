local SparseBlockTemporalConvolution, parent = torch.class('nn.SparseBlockTemporalConvolution', 'nn.Module')

--[[ Module description:
    current implementation assumes: 
    a) fullbatch 
    b) default input is allways zero  which enables for sparse backpropagaion. (teDefault is not used at this point) 
    c) no bias in this layer

    Note: isRelax means to allow this module to act as "identity" for column input in which the width becomes smaller than kW
          It needs to be used carefully knowning the effects. 
          namely, if any given column is affected by relax, changes in frameSize will not be applied to it either.
          therefore watch for "inconsistent tensor size" type of errors in subequent layers 
--]]

function SparseBlockTemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW, isRelax)
   dW = dW or 1

   self.inputFrameSize = inputFrameSize -- # of input channels
   self.outputFrameSize = outputFrameSize -- # of output channels
   self.kW = kW
   self.dW = dW

    self.isRelax = isRelax or false
    

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.dummyBias = torch.zeros(outputFrameSize)
   
   self:reset()
end

function SparseBlockTemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end

   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
   end
end

function SparseBlockTemporalConvolution:pri_ensureOutput(input)
   if self.output ~= nil then
      return
   end


   self.output = { nBatchSize = input.nBatchSize, taData = {} }

   local nColumns = table.getn(input.taData)
   for i=1, nColumns do
      local taInputCurr = input.taData[i]
      
      if self.isRelax and taInputCurr.teValue:size(2) < self.kW then -- when smaller then identity
         taOutputCurr = { teValue = torch.Tensor(), 
                          teRowIdx = torch.LongTensor() }
      else
         local nRows = taInputCurr.teValue:size(1)
         local nWidth =  (taInputCurr.teValue:size(2) - self.kW) / self.dW + 1;

         taOutputCurr = { teValue = torch.zeros(nRows, nWidth, self.outputFrameSize),
                          teRowIdx = taInputCurr.teRowIdx }
      end

      table.insert(self.output.taData, taOutputCurr)
   end

end

function SparseBlockTemporalConvolution:pri_ensureGradInput(input)
   if self.gradInput ~= nil then
      return
   end

   self.gradInput = { nBatchSize = input.nBatchSize, taData = {} }

   local nColumns = table.getn(input.taData)
   for i=1, nColumns do
      local taInputCurr = input.taData[i]

      if self.isRelax and taInputCurr.teValue:size(2) < self.kW then -- when smaller then identity
         taGradInputCurr = { teValue = torch.Tensor(), 
                             teRowIdx = torch.LongTensor() }
      else
         taGradInputCurr = { teValue = torch.zeros( taInputCurr.teValue:size() ),
                             teRowIdx = taInputCurr.teRowIdx }
      end

      table.insert(self.gradInput.taData, taGradInputCurr)
   end
end

function SparseBlockTemporalConvolution:pri_updateOutput_column(taInput, taOutput)

   if self.isRelax and taInput.teValue:size(2) < self.kW then -- when smaller then identity
      taOutput.teValue:set(taInput.teValue)
      taOutput.teRowIdx:set(taInput.teRowIdx)
   else
      local input = taInput.teValue
      local output = taOutput.teValue

      input.THNN.TemporalConvolution_updateOutput(input:cdata(), output:cdata(),
                                                  self.weight:cdata(), self.dummyBias:cdata(),
                                                  self.kW, self.dW,
                                                  self.inputFrameSize, self.outputFrameSize)
   end

end

function SparseBlockTemporalConvolution:updateOutput(input)
   -- ensure output created
   self:pri_ensureOutput(input)

   -- update data
   local nColumns = table.getn(self.output.taData)
   for i=1, nColumns do
      self:pri_updateOutput_column(input.taData[i], 
                                   self.output.taData[i])
   end
   
   return self.output
end

function SparseBlockTemporalConvolution:pri_updateGradInput_column(taInput, taGradOutput, taGradInput)
   -- Note: mimicing the logic of nDimension==3 in "TemporalConvolution_updateGradInput"
   if self.isRelax and taInput.teValue:size(2) < self.kW then -- when smaller then identity
      taGradInput.teValue:set(taGradOutput.teValue)
      taGradInput.teRowIdx:set(taGradOutput.teRowIdx)
   else
      local input = taInput.teValue
      local gradOutput = taGradOutput.teValue
      local gradInput = taGradInput.teValue

      input.THNN.TemporalConvolution_updateGradInput(input:cdata(), gradOutput:cdata(),
                                                     gradInput:cdata(), self.weight:cdata(),
                                                     self.kW, self.dW)
   end

end

function SparseBlockTemporalConvolution:updateGradInput(input, gradOutput)
   -- ensure GradInput created
   self:pri_ensureGradInput(input)

   -- update data
   local nColumns = table.getn(self.gradInput.taData)
   for i=1, nColumns do
      self:pri_updateGradInput_column(input.taData[i],
                                      gradOutput.taData[i],
                                      self.gradInput.taData[i])
   end
   
   return self.gradInput
end

function SparseBlockTemporalConvolution:pri_accGradParameters_column(taInput, taGradOutput, scale)

   if self.isRelax and taInput.teValue:size(2) < self.kW then -- when smaller then identity
      return
   end

   local input = taInput.teValue
   local gradOutput = taGradOutput.teValue
  input.THNN.TemporalConvolution_accGradParameters(input:cdata(), gradOutput:cdata(),
                                                   self.gradWeight:cdata(), self.dummyBias:cdata(),
                                                   self.kW, self.dW, scale)

end

function SparseBlockTemporalConvolution:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1

   local nColumns = table.getn(input.taData)
   for i=1, nColumns do
      self:pri_accGradParameters_column(input.taData[i],
                                        gradOutput.taData[i],
                                        scale)
   end

end

