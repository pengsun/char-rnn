-- simple script that loads a checkpoint and prints its opts

require 'torch'
require 'nn'
require 'nngraph'

require 'util.OneHot'
require 'util.misc'
require 'util.env_utils'
model_utils = require 'util.model_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Load a checkpoint and print its options and validation losses.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','model to load')
cmd:option('-gpuid',0,'gpu to use')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

setup_env()

local checkpoint = torch.load(opt.model)

local function print_kv(t, k)
  print(k .. ' = ')
  print(t[k])
end

print_kv(checkpoint, 'opt')

print_kv(checkpoint, 'val_losses')

print_kv(checkpoint, 'vocab')
--print('#vocab = '); print(table_len(checkpoint.vocab))

params, _ = model_utils.combine_all_parameters(checkpoint.protos.rnn)
print('#params = '); print(params:numel())
