--[[ test on a sequence and calculate bits-per-char]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'xlua'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

-- global variables
opt = nil
protos = nil
init_state = nil

function is_cu()
  return opt.gpuid >= 0 and opt.opencl == 0
end

function is_cl()
  return opt.gpuid >= 0 and opt.opencl == 1
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

function get_split_sizes()
  local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
  return {opt.train_frac, opt.val_frac, test_frac}
end

function is_vocab_compatible(v1, v2)
  local vocab_compatible = true
  for c,i in pairs(v2) do 
    if not v1[c] == i then 
      vocab_compatible = false
    end
  end
  return vocab_compatible
end

function assert_saved_vocab_compatible(v, saved_v)
  if not is_vocab_compatible(v, saved_v) then
    error('the character vocabulary for this dataset and the one' .. 
      'in the saved checkpoint are not the same. This is trouble.')
  end  
end

function load_model_checkpoint()
  if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist.' .. 
      'Are you sure you didn\'t forget to prepend cv/ ?')
  end
  return torch.load(opt.model)
end

function create_clones(prototypes, seq_length)
  -- make a bunch of clones after flattening, as that reallocates memory
  local clones = {}
  for name,proto in pairs(prototypes) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, seq_length, not proto.parameters)
  end
  return clones
end

function get_prototypes(checkpoint)
  protos = checkpoint.protos
  protos.rnn:evaluate() -- put in eval mode so that dropout works properly
  return protos
end

function get_states(checkpoint)
  -- initialize the rnn state to all zeros
  print('creating an ' .. checkpoint.opt.model .. '...')
  print('num_layers = ' .. checkpoint.opt.num_layers)
  print('rnn_size = ' .. checkpoint.opt.rnn_size)
  
  local current_state = {}
  for L = 1, checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(opt.batch_size, checkpoint.opt.rnn_size):double()
    table.insert(current_state, h_init:clone())
    
    if checkpoint.opt.model == 'lstm' then
      table.insert(current_state, h_init:clone())
    end
  end
  return current_state
end

function main()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a character-level language model')
  cmd:text()
  cmd:text('Options')
  -- required: saved model path
   cmd:argument('-model','model checkpoint to use for sampling')
  -- dataset & its partition
  cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
  cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
  cmd:option('-val_frac',0.00,'fraction of data that goes into validation set') -- test_frac will be computed as (1 - train_frac - val_frac)
  -- parameters
  cmd:option('-seq_length',54,'number of timesteps to unroll for')
  cmd:option('-batch_size',48,'number of sequences to train on in parallel')
  -- GPU/CPU
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:text()
  -- parse input params
  opt = cmd:parse(arg)
  
  setup_env()
  
  -- create the data loader class
  local split_sizes = get_split_sizes() -- {train, validate, test} in percentage
  require('mobdebug').start()
  local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
  print('batch_size = ' .. loader.batch_size)
  print('vocab size: ' .. loader.vocab_size)
  
  -- load models
  local checkpoint = load_model_checkpoint()
  assert_saved_vocab_compatible(loader.vocab_mapping, checkpoint.vocab)
  local protos = get_prototypes(checkpoint)
  local init_state = get_states(checkpoint)

  -- ship the model and state to the GPU if desired
  model_table_togpu(protos)
  tensor_table_togpu(init_state)
  
  -- put into one flattened parameters tensor
  params, grad_params = model_utils.combine_all_parameters(protos.rnn)
  print('number of parameters in the model: ' .. params:nElement())
  
  -- unroll along the time axis
  local clones = create_clones(protos, opt.seq_length)
  
  -- help function
  function eval_split(split_index, rnn_state)
    -- evaluate the loss over an entire split
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    rnn_state = rnn_state or {[0] = init_state}

    for i = 1, n do -- iterate over batches in the split
      -- fetch a batch
      local x, y = loader:next_batch(split_index)
      x, y = xy_togpu(x, y)
      -- forward pass
      for t = 1, opt.seq_length do
        clones.rnn[t]:evaluate() -- for dropout proper functioning
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
        prediction = lst[#lst] 
        loss = loss + clones.criterion[t]:forward(prediction, y[t])
      end
      -- carry over lstm state
      rnn_state[0] = rnn_state[#rnn_state]
      -- progress bar
      xlua.progress(i, n)
    end

    loss = loss / opt.seq_length / n
    return loss, rnn_state
  end -- eval_split

  function loss_to_bitsperchar(loss)
    return loss / math.log(2)
  end
  
  -- warming up by going through the the traning set
  print('warm up on training set...')
  local train_loss, rnn_state = eval_split(1) -- 1 for training set
  local train_bpc = loss_to_bitsperchar(train_loss)
  print('train loss = ' .. train_loss)
  print('train bpc = ' .. train_bpc)
  print('\n')
  
  -- do the real evaluation on testing set
  print('evaluate on testing set...')
  local test_loss, _ = eval_split(3, rnn_state) -- 3 for training set
  local test_bpc = loss_to_bitsperchar(test_loss)
  print('test loss = ' .. test_loss)
  print('test bpc = ' .. test_bpc)
  print('\n')
  
end

main()