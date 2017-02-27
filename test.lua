require 'torch'
require 'loader'
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')
local cmd = torch.CmdLine()
cmd:option('-mode', 'test', 'Test mode')
cmd:option('-pretrained', true, 'Load pretrained weights')
cmd:option('-modelName', './models/epoch20.t7', 'Path to pre-trained model to load')
cmd:option('-backend', 'nn', 'Set to cudnn to use GPU')
cmd:option('-batchSize', 128, 'Batch size in training')
cmd:option('-numHidden', 6, 'Number of hidden layers')
cmd:option('-config', 'config.lua', 'Configuration file containing architecture params')
cmd:text()

local opt = cmd:parse(arg)
print(opt)
local cfg = paths.dofile(opt.config)

if opt.backend == 'nn' then
  backend = nn
else
  require 'cudnn'
  backend = cudnn
end

local function test()
  local model = torch.load(opt.modelName)
  model:remove() -- replace LogSoftmax with Softmax
  model:add(backend.SoftMax())
  print("Retrieving test set...")
  local loader = Loader.new(model, cfg, "test")
  local X,y = loader:getDataset()
  print("Loaded test set of "..X:size(1).." instances.")

  local num = X:size(1)
  correct = num
  scores = {0,0,0,0,0}
  totals = {0,0,0,0,0}
  print("Testing network...")
  for i = 1, num do
    local output = model:forward(X[i]:view(1,X[i]:size(1),X[i]:size(2)))
    k,class = torch.max(output,1)
    totals[class[1]] = totals[class[1]]+1
    if class[1] ~= y[i] then
      scores[class[1]] = scores[class[1]] + 1
      correct = correct-1
    end
  end
  print("% Overall Accuracy: "..(correct/num)*100.0)
  print("% Accuracies By Genre: ")
  for j=1, #scores do
    print(cfg.genres[j].." score: "..((totals[j]-scores[j])/totals[j])*100)
  end
end

test()
