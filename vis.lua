
--[[ This file samples characters from a trained model

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
require 'util.env_utils'

function gprint(str)
  -- gated print: simple utility function wrapping a print
  if opt.verbose == 1 then print(str) end
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

function pred_on_seq(seq_text, cur_state, vocab, ivocab)
  local states = {}
  local state_size = #cur_state
  local prediction
  for i = 1, #seq_text do
    c = seq_text:sub(i,i)
    local c_ind = vocab[c]
    if not c_ind then error('char ' .. c .. ' not in vocab') end
    cur_char = torch.Tensor{c_ind}
    if is_cu() then cur_char = cur_char:cuda() end
    if is_cl() then cur_char = cur_char:cl() end
    
    -- outputs: a list of [state1,state2,..stateN,prediction]. 
    local outputs = protos.rnn:forward{cur_char, unpack(cur_state)}
    
    -- update current state
    cur_state = {}
    for j = 1, state_size do table.insert(cur_state, outputs[j]) end
    -- update states over time
    states[i] = clone_list(cur_state)
    -- prediction
    prediction = outputs[#outputs] -- last element holds the log probabilities
  end
  return prediction, cur_state, states
end

function pred_random(size)
  local prediction = torch.Tensor(1, size):fill(1)/(size)
  if is_cu() then prediction = prediction:cuda() end
  if is_cl() then prediction = prediction:cl() end
  return prediction
end

function rand_quotestr(slen, vocab, ivocab)
  local lenTotal = 2 + slen.left + slen.mid + slen.right
  local seq = ''
  local vocab_size = table_len(ivocab)
  local i1, i2 = slen.left+1, slen.left+slen.mid+2
  
  for i = 1, lenTotal do
    if i == i1 or i == i2 then -- quote
      seq = seq .. '"'
    else -- random character
      local num
      --print('i' .. i)
      while true do -- any char other than a quote
        num = torch.ceil( torch.uniform(0, vocab_size) )
        --print('num = ' .. num)
        if num ~= vocab['"'] then break end
      end
      seq = seq .. ivocab[num]
    end
  end -- for i
  return seq
end

function gen_betweenquote_tar(text)
  local tar = -torch.ones(#text):float()
  local flag_between = false
  for i = 1, #text do
    local c = text:sub(i,i)
    if c == '"' then 
      flag_between = not flag_between
    end
    if flag_between then tar[i] = 1 end
  end
  return tar
end

function states_to_tensor(states)
  -- states: [t][2*ell] list, each [1][u] tensor
  -- ts: [ell][u][t]
  local T = table_len(states)
  local L = table_len(states[1]) / 2
  local U = states[1][1]:numel()
  
  local ts = torch.Tensor(L,U,T):type(states[1][1]:type())
  for ell = 1, L do
    for u = 1, U do
      for t = 1, T do
        ts[ell][u][t] = states[t][2*ell][1][u]
      end
    end
  end
  
  return ts
end

function find_strong_act_lstm(tar, states)
  -- states: [t][2*ell] list, each [1][u] tensor
  local ts = states_to_tensor(states)
  local i_lay, i_unit, act
  local d = 0
  local nt = tar:norm()
  for ell = 1, ts:size(1) do
    for u = 1, ts:size(2) do
      local cur_act = ts[ell][u]
      
      -- dist
--      local cur_d = torch.dist(cur_act, tar)
--      if cur_d < d then
--        d = cur_d
--        i_lay, i_unit, act = ell, u, cur_act:clone()
--      end 
      
      -- cos(angle)
      local cur_d = torch.dot(cur_act, tar)
      cur_d = cur_d / (cur_act:norm() * nt)

      if cur_d > d then
        d = cur_d
        i_lay, i_unit, act = ell, u, cur_act:clone()
      end 
    end
  end
  return i_lay, i_unit, act
end

function threshold_activation(act)
  act = act:mul(10)
  nn = require'nn'
  local tf = nn.Tanh():type(act:type())
  return tf:forward(act)
end


function get_opt()
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Visualize')
  cmd:text()
  cmd:text('Options')
  -- required:
  cmd:argument('-model','model checkpoint to use for sampling')
  -- optional parameters
  cmd:option('-seed',123,'random number generator\'s seed')
  cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
  cmd:option('-text',"","the string to be investigated.")
  cmd:option('-thres_act',0,"threshold the activation when plotting.")
  cmd:option('-show_baseline',0,"wheter to show baseline curve.")
  cmd:option('-show_tar',0,"wheter to show target curve.")
  cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
  cmd:text()
  opt = cmd:parse(arg)
  return opt
end

function main ()
  opt = get_opt()
  setup_env()
  
  local checkpoint = load_model_checkpoint()
  local protos = get_prototypes(checkpoint)
  local cur_state = get_states(checkpoint)
  
  -- vocabulary
  local vocab = checkpoint.vocab
  local ivocab = make_inverse_vocabulary(vocab)
  local vocab_ascii = make_vocabulary_letter()
  local ivocab_ascii = make_inverse_vocabulary(vocab_ascii)
  
  -- do a few seeded timesteps
  local cur_pred
  local seed_text = opt.primetext
  if string.len(seed_text) > 0 then
    print('seeding with: ')
    print(seed_text)
    cur_pred, cur_state, _ = pred_on_seq(seed_text, cur_state, vocab, ivocab)
  end

  -- generate quoted string
  print('the specific string: ')
  local text
  if string.len(opt.text) == 0 then
    local slen = {left = 12, mid = 22, right = 8}
    text = rand_quotestr(slen, vocab_ascii, ivocab_ascii)
  else
    text = opt.text
  end

  print(text)
  
  -- fprop on i
  print('fprop on the text')
  local states
  _, cur_state, states = pred_on_seq(text, cur_state, vocab, ivocab)
  
  -- find max cell activation
  print('find max cell activation')
  local tar = gen_betweenquote_tar(text)
  tar = togpu(tar)
  if checkpoint.opt.model ~= 'lstm' then 
    error('not an lstm')
  end
  local i_lay, i_unit, max_act = find_strong_act_lstm(
    tar, states)
  
  -- cell at specific layer and unit
  local j_layer, j_unit = 2, 49
  local ts = states_to_tensor(states)
  local act = ts[j_layer][j_unit]:clone()
  
  -- draw
  print('i_layer = ' .. i_lay)
  print('i_unit = ' .. i_unit)
  require'gnuplot'
  if opt.thres_act > 0 then
    max_act = threshold_activation(max_act)
  end
  local what_draw = { {'max-act',max_act,'-'} }
  if opt.show_tar > 0 then 
    table.insert(what_draw, {'tar',tar,'+'}) 
  end
  if opt.show_baseline > 0 then 
    table.insert(what_draw, {'baseline-act',act,'-'})
  end
  gnuplot.plot(unpack(what_draw))
end

main()
