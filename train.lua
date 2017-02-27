require 'torch'
require 'nn'
require 'optim'
require 'paths'
require 'loader'
torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:option('-weights', false, 'Previously saved model weights to load')
cmd:option('-mode', 'train', 'Training mode')
cmd:option('-modelFactory', 'model.lua', 'Lua file to generate model definition')
cmd:option('-backend', 'nn', 'Set to cudnn to use GPU')
cmd:option('-logsTrainPath', './logs/training/', ' Path to save Training logs')
cmd:option('-logsValPath', './logs/val/', ' Path to save Validation logs')
cmd:option('-epochSave', false, 'save model every epoch')
cmd:option('-trainPath', './models/', ' Path to save model between epochs')
cmd:option('-saveName', 'deepgenre.t7', 'Name of serialized model')
cmd:option('-epochs', 20, 'Number of epochs for training')
cmd:option('-learningRate', 0.001, 'Training learning rate')
cmd:option('-classes', 5, 'Number of genres to classify')
cmd:option('-config', 'config.lua', 'Configuration file containing architecture params')
cmd:option('-batchSize', 1, 'Batch size in training')
-- cmd:option('-numHidden', 6, 'Number of hidden layers')
cmd:text()
local opt = cmd:parse(arg)
print(opt)
local backend
if opt.backend == 'nn' then
  backend = nn
else
  require 'cudnn'
  backend = cudnn
end
local cfg = paths.dofile(opt.config)
local sliceAudio = paths.dofile('data.lua').sliceAudio
local rename = paths.dofile('data.lua').labelGenres
--utils = paths.dofile('utils.lua')

function train(model, lr, createSpectrograms)
  --if rename then rename(cfg.dir.data) end
  if createSpectrograms then sliceAudio(cfg) end

  print("Preparing dataset...")
  local loader = Loader.new(model, cfg, 'train')
  local X,y = loader:getDataset()
  print("Dataset prepared!")
  print("Training....")
  x, gradParams = model:getParameters()
  local criterion = backend.ClassNLLCriterion()
  criterion.sizeAverage = false
  local optimState = {learningRate = lr}
  local currentLoss
  local startTime = os.time()

  -- training
  local currentLoss
  local startTime = os.time()
  for i=1, opt.epochs do
    local averageLoss = 0
    for j = 1, X:size(1) do
      -- function to give to optim
      feval = function(x)
         gradParams:zero()
         input = X[j]:clone()
         label = y[j]
         local unsqueeze = backend.Unsqueeze(1) -- unsqueeze mini-batch dim (1x128x128)
         input = unsqueeze:forward(input)
         local output = model:forward(input)
         local loss = criterion:forward(output, label)
         print("iter: ", j, "loss: ", loss)
         local gradOutput = criterion:backward(output, label)
         model:backward(input, gradOutput)
         return loss, gradParams
      end
        currentLoss = 0
        local _, fs = optim.rmsprop(feval, x, optimState)
        currentLoss = currentLoss + fs[1]
        print(currentLoss)
        --xlua.progress(j, self.indexer.nbOfBatches)
        averageLoss = averageLoss + currentLoss
    end
    averageLoss = averageLoss / X:size(1)
    print('loss after epoch: ', i, 'is: ', averageLoss)
    if opt.epochSave then torch.save(opt.trainPath..'epoch'..i..'.t7', model) end
  end
  torch.save(opt.trainPath..opt.saveName,model)
end

local modelFactory = paths.dofile(opt.modelFactory)
local model = modelFactory(backend, opt.classes, opt.batchSize, opt.mode)
train(model, 0.001, false)
