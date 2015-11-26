
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.OneHot'
require 'util.misc'

function gprint(str)
  -- gated print: simple utility function wrapping a print
  if opt.verbose == 1 then print(str) end
end

function is_cu()
  return opt.gpuid >= 0 and opt.opencl == 0
end

function is_cl()
  return opt.gpuid >= 0 and opt.opencl == 1
end

function setup_env()
  -- check that cunn/cutorch are installed if user wants to use the GPU
  if is_cu() then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
      gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
      gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
      cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      cutorch.manualSeed(opt.seed)
    else
      gprint('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

  -- check that clnn/cltorch are installed if user wants to use OpenCL
  if is_cl() then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
      gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
      gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
      cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
      torch.manualSeed(opt.seed)
    else
      gprint('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end

  -- for random number
  torch.manualSeed(opt.seed)
end

function load_model_checkpoint()
  if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist.' .. 
      'Are you sure you didn\'t forget to prepend cv/ ?')
  end
  return torch.load(opt.model)
end

function get_prototypes(checkpoint)
  protos = checkpoint.protos
  protos.rnn:evaluate() -- put in eval mode so that dropout works properly
  return protos
end

function make_inverse_vocabulary(vocab)
  local ivocab = {}
  for c,i in pairs(vocab) do ivocab[i] = c end
  return ivocab
end

function get_states(checkpoint)
  -- initialize the rnn state to all zeros
  gprint('creating an ' .. checkpoint.opt.model .. '...')
  local current_state = {}
  for L = 1, checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
      table.insert(current_state, h_init:clone())
    end
  end
  return current_state
end


function pred_on_seq(seq_text, states, vocab)
  local prediction
  for c in seq_text:gmatch'.' do
    cur_char = torch.Tensor{vocab[c]}
    io.write(ivocab[cur_char[1]])
    if is_cu() then cur_char = cur_char:cuda() end
    if is_cl() then cur_char = cur_char:cl() end
    
    -- outputs: a list of [state1,state2,..stateN,prediction]. 
    local outputs = protos.rnn:forward{cur_char, unpack(states)}
    
    states = {}
    for i = 1, state_size do table.insert(states, outputs[i]) end
    prediction = outputs[#outputs] -- last element holds the log probabilities
  end
  return prediction
end

function pred_random(size)
  local prediction = torch.Tensor(1, size):fill(1)/(size)
  if is_cu() then prediction = prediction:cuda() end
  if is_cl() then prediction = prediction:cl() end
  return prediction
end

function gen_seq(protos, current_state, current_prediction, ivocab)
  seq = ''
  local state_size = #current_state
  for i = 1, opt.length do

    -- log probabilities from the previous timestep
    if opt.sample == 0 then
      -- use argmax
      local _, prev_char_ = current_prediction:max(2)
      cur_char = prev_char_:resize(1)
    else
      -- use sampling
      current_prediction:div(opt.temperature) -- scale by temperature
      local probs = torch.exp(current_prediction):squeeze()
      probs:div(torch.sum(probs)) -- renormalize so probs sum to one
      cur_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- forward the rnn for next character
    local outputs = protos.rnn:forward{cur_char, unpack(current_state)}
    -- first N-1 elements hold states
    current_state = {}
    for i = 1, state_size do 
      table.insert(current_state, outputs[i]) 
    end
    -- last element holds the log probabilities
    current_prediction = outputs[#outputs] 

    seq = seq .. ivocab[cur_char[1]]
    io.write(string.sub(seq, -1, -1))
  end
  return seq
end

function main ()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Sample from a character-level language model')
  cmd:text()
  cmd:text('Options')
  -- required:
  cmd:argument('-model','model checkpoint to use for sampling')
  -- optional parameters
  cmd:option('-seed',123,'random number generator\'s seed')
  cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
  cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
  cmd:option('-length',2000,'number of characters to sample')
  cmd:option('-temperature',1,'temperature of sampling')
  cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
  cmd:text()
  opt = cmd:parse(arg)

  setup_env()

  local checkpoint = load_model_checkpoint()
  local protos = get_prototypes(checkpoint)
  local vocab = checkpoint.vocab
  local ivocab = make_inverse_vocabulary(vocab)
  local cur_state = get_states(checkpoint)
  
  -- do a few seeded timesteps
  local cur_pred
  local seed_text = opt.primetext
  if string.len(seed_text) > 0 then
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
    cur_pred = pred_on_seq(seed_text, cur_state, vocab)
  else
    -- fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, ' .. 
      'using uniform probability over first character')
    gprint('--------------------------')
    cur_pred = pred_random(#ivocab)
  end

  -- start sampling/argmaxing
  local seq = gen_seq(protos, cur_state, cur_pred, ivocab)
  io.write('\n') 
  io.flush()
end

main()

