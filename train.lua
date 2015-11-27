--[[ This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

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

function load_model()
  print('loading an LSTM from checkpoint ' .. opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  
  -- overwrite model settings based on checkpoint to ensure compatibility
  print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. 
    ', num_layers=' .. checkpoint.opt.num_layers .. 
    ' based on the checkpoint.')
  opt.rnn_size = checkpoint.opt.rnn_size
  opt.num_layers = checkpoint.opt.num_layers

  return checkpoint.protos, checkpoint.vocab
end

function create_prototypes(vocab_size)
  print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
  local protos = {}
  if opt.model == 'lstm' then
    protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
  elseif opt.model == 'gru' then
    protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
  elseif opt.model == 'rnn' then
    protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
  end
  protos.criterion = nn.ClassNLLCriterion()
  return protos
end

function initialize_state()
  local init_state = {}
  for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
      table.insert(init_state, h_init:clone())
    end
  end
  return init_state
end

function initialize_parameters(prototypes)
  local params, _ = model_utils.combine_all_parameters(prototypes.rnn)
  
  -- initialization
  params:uniform(-0.08, 0.08) -- small uniform numbers
  
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if opt.model == 'lstm' then
    for layer_idx = 1, opt.num_layers do
      for _,node in ipairs(prototypes.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
          node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
        end
      end -- for node
    end -- for layer_idx
  end -- if
end

function main()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a character-level language model')
  cmd:text()
  cmd:text('Options')
  -- data
  cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
  -- model params
  cmd:option('-rnn_size', 128, 'size of LSTM internal state')
  cmd:option('-num_layers', 2, 'number of layers in the LSTM')
  cmd:option('-model', 'lstm', 'lstm,gru or rnn')
  -- optimization
  cmd:option('-learning_rate',2e-3,'learning rate')
  cmd:option('-learning_rate_decay',0.97,'learning rate decay')
  cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
  cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
  cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-seq_length',50,'number of timesteps to unroll for')
  cmd:option('-batch_size',50,'number of sequences to train on in parallel')
  cmd:option('-max_epochs',50,'number of full passes through the training data')
  cmd:option('-grad_clip',5,'clip gradients at this value')
  cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
  cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
              -- test_frac will be computed as (1 - train_frac - val_frac)
  cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
  -- bookkeeping
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
  cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
  cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
  cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
  cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
  -- GPU/CPU
  cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:text()
  -- parse input params
  opt = cmd:parse(arg)
  
  setup_env()
  
  -- create the data loader class
  local split_sizes = get_split_sizes() -- {train, validate, test} in percentage
  local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
  print('vocab size: ' .. loader.vocab_size)
  
  -- make sure output directory exists
  if not path.exists(opt.checkpoint_dir) then 
    lfs.mkdir(opt.checkpoint_dir) 
  end

  -- define the model: prototypes for one timestep, then clone them in time
  local do_random_init = true
  if string.len(opt.init_from) > 0 then
    do_random_init = false
    protos, saved_vocab_mapping = load_model()
    assert_saved_vocab_compatible(loader.vocab_mapping, saved_vocab_mapping)
  else
    protos = create_prototypes(loader.vocab_size)
  end
  
  -- initialize all the internal parameters & the cell/hidden states
  if do_random_init then
    initialize_parameters(protos)
  end
  init_state = initialize_state()

  -- ship the model and state to the GPU if desired
  model_table_togpu(protos)
  tensor_table_togpu(init_state)
  
  -- put into one flattened parameters tensor
  params, grad_params = model_utils.combine_all_parameters(protos.rnn)
  print('number of parameters in the model: ' .. params:nElement())
  
  local clones = model_utils.create_clones(protos, opt.seq_length)
  local init_state_global = clone_list(init_state)
  
  -- evaluate the loss over an entire split
  function eval_split(split_index)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}

    for i = 1,n do -- iterate over batches in the split
      -- fetch a batch
      local x, y = loader:next_batch(split_index)
      x,y = xy_togpu(x,y)
      -- forward pass
      for t=1,opt.seq_length do
        clones.rnn[t]:evaluate() -- for dropout proper functioning
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
        prediction = lst[#lst] 
        loss = loss + clones.criterion[t]:forward(prediction, y[t])
      end
      -- carry over lstm state
      rnn_state[0] = rnn_state[#rnn_state]
      print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
  end -- eval_split

  -- do fwd/bwd and return loss, grad_params
  function feval(x)
    if x ~= params then
      params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    x,y = xy_togpu(x,y)
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
      clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      predictions[t] = lst[#lst] -- last element is the prediction
      loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
      -- backprop through loss, and softmax/linear
      local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
      table.insert(drnn_state[t], doutput_t)
      local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k > 1 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          drnn_state[t-1][k-1] = v
        end
      end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
  end -- feval

  -- start optimization here
  local train_losses, val_losses = {}, {} -- losses over iterations
  local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
  
  local iterations, iterations_per_epoch = opt.max_epochs*loader.ntrain, loader.ntrain
  
  for i = 1, iterations do
    local time = 0
    local loss = nil -- current loss
    local epoch = i / loader.ntrain
    
    local function decay_learning_rate()
      if epoch >= opt.learning_rate_decay_after then
        local decay_factor = opt.learning_rate_decay
        optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
        print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
      end
    end
    
    local function save_checkpoint()
      -- evaluate loss on validation data
      local val_loss = eval_split(2) -- 2 = validation
      val_losses[i] = val_loss

      local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.protos = protos
      checkpoint.opt = opt
      checkpoint.train_losses = train_losses
      checkpoint.val_loss = val_loss
      checkpoint.val_losses = val_losses
      checkpoint.i = i
      checkpoint.epoch = epoch
      checkpoint.vocab = loader.vocab_mapping
      torch.save(savefile, checkpoint)
    end
    
    local function print_progress()
      tmpl = '%d/%d (epoch %.3f), ' .. 
        'train_loss = %6.8f, grad/param norm = %6.4e, ' .. 
        'time/batch = %.4fs'
      print(string.format(tmpl, 
          i, iterations, epoch, 
          train_losses[i], grad_params:norm() / params:norm(), 
          time))
    end
    
    local function should_stop()
      local flag, info = false, ''
      -- nan error ?
      if loss[1] ~= loss[1] then
        flag = true
        info = 'loss is NaN.  This usually indicates a bug.  ' .. 
        'Please check the issues page for existing issues, or create a new issue, if none exist.  ' .. 
        'Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?'
      end
      -- exploding
      if train_losses[i] > train_losses[1] * 3 then
        flag = true
        info = 'loss is exploding, aborting.'
      end
      return flag, info
    end
    
    -- fprop & bprop timing
    local timer = torch.Timer() ----------------------------------------------------------------
    _, loss = optim.rmsprop(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0 then
      cutorch.synchronize()
    end
    time = timer:time().real--------------------------------------------------------------------

    -- bookkeep
    train_losses[i] = loss[1] -- the loss is inside a list, pop it

    -- exponential weight decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
      decay_learning_rate()
    end

    -- save every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
      save_checkpoint()
    end

    if i % opt.print_every == 0 then
      print_progress()
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    flag, info = should_stop()
    if flag then 
      print(info)
      break
    end -- if flag
    
  end -- for i
end

main()
