--[[ visualize cell activation ]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'xlua'

require 'util.OneHot'
require 'util.misc'
require 'util.env_utils'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

-- global variables
opt = nil
protos = nil
init_state = nil

function get_split_sizes()
  local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
  return {opt.train_frac, opt.val_frac, test_frac}
end

function load_model_checkpoint()
  if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist.' .. 
      'Are you sure you didn\'t forget to prepend cv/ ?')
  end
  return torch.load(opt.model)
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

function rand_quotestr(slen, vocab)
  local lenTotal = 2 + slen.left + slen.mid + slen.right
  local seq = torch.zeros(lenTotal):float()
  -- random
  --print('vocab size = ' .. table_len(vocab))
  --print('vocab quote = ' .. vocab['"'])
  for i = 1, lenTotal do
    local num
    --print('i' .. i)
    while true do -- any char other than a quote
      num = torch.ceil( torch.uniform(0, table_len(vocab)) )
      --print('num = ' .. num)
      if num ~= vocab['"'] then break end
    end
    seq[i] = num
  end
  -- quote
  seq[slen.left+1] = vocab['"']
  seq[slen.left+slen.mid+2] = vocab['"']
  return seq
end

function gen_betweenquote_tar(slen)
  local lenTotal = 2 + slen.left + slen.mid + slen.right
  local tar = torch.zeros(slen)
  local first, last = slen.left+2, slen.left+slen.mid+1
  for i = first, last do
    tar[i] = 1
  end
  return tar
end

function find_strong_act(tar, states)

  return i_lay, i_unit, act
end


function get_opt()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('visualize cell activation')
  cmd:text()
  cmd:text('Options')
  -- required: saved model path
  cmd:argument('-model','model checkpoint to use for sampling')
  -- dataset & its partition
  cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
  cmd:option('-primetext_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
  cmd:option('-train_frac',0.0001,'fraction of data that goes into train set')
  cmd:option('-val_frac',0.00,'fraction of data that goes into validation set') -- test_frac will be computed as (1 - train_frac - val_frac)
  -- parameters
  cmd:option('-seq_length',100,'number of timesteps to unroll for')
  cmd:option('-batch_size',200,'number of sequences to train on in parallel')
  -- GPU/CPU
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:text()
  -- parse input params
  opt = cmd:parse(arg)
  
  return opt
end


function main()
  opt = get_opt()
  
  setup_env()
  
  -- create the data loader class
  local split_sizes = get_split_sizes() -- {train, validate, test} in percentage
  local loader = CharSplitLMMinibatchLoader.create(opt.primetext_dir,
    opt.batch_size, opt.seq_length, split_sizes)
  print('batch_size = ' .. loader.batch_size)
  
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
  local clones = model_utils.create_clones(protos, opt.seq_length)
  
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

  function fprop_seq(x, rnn_state)
    -- fprop over a sequence
    print('evaluating sequence...')
    local loss = 0
    assert(rnn_state, 'rnn_state not passed')
    x = togpu(x)
    
    -- forward pass
    for t = 1, opt.seq_length do
      clones.rnn[t]:evaluate() -- for dropout proper functioning
      local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
      
      rnn_state[t] = {}
      for i = 1, #init_state do 
        table.insert(rnn_state[t], lst[i]) 
      end
      --prediction = lst[#lst] 
    end
    -- carry over lstm state
    rnn_state[0] = rnn_state[#rnn_state]

    return rnn_state
  end -- eval_split  

  function loss_to_bitsperchar(loss)
    return loss / math.log(2)
  end
  
  -- warming up by going through the the traning set
  print('warm up on training set...')
  require'mobdebug'.start()
  local train_loss, rnn_state = eval_split(1) -- 1 for training set
  local train_bpc = loss_to_bitsperchar(train_loss)
  print('train loss = ' .. train_loss)
  print('train bpc = ' .. train_bpc)
  print('\n')
  
  -- 
  print('generating random quote sequence...')
  local slen = {left = 5, mid = 22, right = 8}
  local x = rand_quotestr(slen, checkpoint.vocab)
  togpu(x)
  make_rowvector(x)
  
  -- fprop
  print('fprop on sequence...')
  require'mobdebug'.start()
  rnn_state = fprop_seq(x, rnn_state)
  
  -- find the cell sensitive to specific event
  print('finding the sensitive cell...') 
  local y = quotestr_to_tar(x)
  y = togpu(y)
  local i_lay, i_unit, cell_act = find_strong_act(
    y, rnn_state)
  
  -- draw
  print('#layer = ' .. i_lay)
  print('#unit = ' .. i_unit)
  
  print('y = ')
  print(y)
  
  print('cell activation = ')
  print(cell_act)
  
end

main()