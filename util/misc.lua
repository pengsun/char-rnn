--[[ miscellaneous utilities ]]--

function table_len(t)
  local count = 0
  for _ in pairs(t) do count = count + 1 end
  return count
end

function clone_list(tensor_list, zero_too)
  -- utility function. todo: move away to some utils file?
  -- takes a list of tensors and returns a list of cloned tensors
  local out = {}
  for k,v in pairs(tensor_list) do
    out[k] = v:clone()
    if zero_too then out[k]:zero() end
  end
  return out
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

function make_vocabulary_ascii()
  local str = "abcdefghijklmnopqrstuvwxyz0123456789" .. 
    "-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  local vocab = {}
  for i = 1, #str do
    vocab[str:sub(i,i)] = i
  end
  return vocab
end

function make_vocabulary_letter()
  local str = "abcdefghijklmnopqrstuvwxyz"
  local vocab = {}
  for i = 1, #str do
    vocab[str:sub(i,i)] = i
  end
  return vocab
end

function make_inverse_vocabulary(vocab)
  local ivocab = {}
  for c,i in pairs(vocab) do ivocab[i] = c end
  return ivocab
end