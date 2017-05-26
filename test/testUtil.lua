do
	local testUtil = {}

	function testUtil.SparseBlockPrint(taInput)
		local taData = taInput.taData
		local nDim = 3 

		if taInput.teDefault ~= nil then
			nDim = taInput.teDefault:dim()
			print("teDefault:" .. tostring(taInput.teDefault:squeeze(3)))
		end

		for i=1, #taData do
			
			nDim = taData[i].teValue:dim()
			io.write(string.format("#%d(non-sparse): \n", i))
			print(taData[i].teValue:squeeze(nDim))

			if taData[i].teDefault ~= nil then
				io.write(string.format("#%d(default): \n", i))
				print(taData[i].teDefault)
			end

		end
	end

   function testUtil.SparseBlockClone(taInput)
      local taRes = { nBatchSize = taInput.nBatchSize , taData = {}}
      for k, taV in pairs(taInput.taData) do
         taRes.taData[k] = { teRowIdx = taV.teRowIdx:clone(), 
                             teValue = taV.teValue:clone()}
      end

      return taRes
   end

	function testUtil.SparseBlockApply(taInput, fuOp)
		if taInput.teDefault ~= nil then
			fuOp(taInput.teDefault)
		end

		local taData = taInput.taData
		for i=1, #taData do
			fuOp(taData[i].teValue)
		end

      return taInput
	end

	function testUtil.getCopyRandomizedBlocks(taInput)
		local taRes = { taData = {}}

		if taInput.teDefault ~= nil then
			taRes.teDefault = torch.rand(taInput.teDefault:size())
		end

		local taData = taInput.taData
		for i=1, #taData do
			local taNew = { teValue = torch.rand(taData[i].teValue:size()),
                         teRowIdx = taData[i].teRowIdx:clone()}
			table.insert(taRes.taData, taNew)
		end


		return taRes
	end

	return testUtil
end
