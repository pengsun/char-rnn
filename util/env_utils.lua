--[[ environment (cpu, cuda, opencl) utilities ]]--

-- global variables
opt = nil

function is_cu()
  return opt.gpuid >= 0 and opt.opencl == 0
end

function is_cl()
  return opt.gpuid >= 0 and opt.opencl == 1
end

function setup_env()
  -- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
  if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

  -- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
  if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
      print('using OpenCL on GPU ' .. opt.gpuid .. '...')
      cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      torch.manualSeed(opt.seed)
    else
      print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
      print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end
  
  torch.manualSeed(opt.seed)
end

function model_table_togpu(t)
  if is_cu() then
    for k,v in pairs(t) do v:cuda() end
  end
  if is_cl() then
    for k,v in pairs(t) do v:cl() end
  end
end

function tensor_table_togpu(t)
  if is_cu() then
    for k,v in pairs(t) do t[k] = v:float():cuda() end
  end
  if is_cl() then
    for k,v in pairs(t) do t[k] = v:cl() end
  end
end

function xy_togpu(x,y)
  x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
  y = y:transpose(1,2):contiguous()
  if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
  end
  if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
    x = x:cl()
    y = y:cl()
  end
  return x,y
end