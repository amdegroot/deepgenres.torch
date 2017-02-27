
require 'nn'
require 'optim'
require 'xlua'
require 'paths'
require 'loader'
local cfg = dofile './config.lua'
local sliceAudio = paths.dofile('songLoader.lua').sliceAudio
local rename = paths.dofile('songLoader.lua').labelGenres
--utils = dofile 'utils.lua'



function train(model, lr, createSpectrograms)
  --if rename then rename(cfg.dir.data) end
  --if createSpectrograms then sliceAudio(cfg) end

  print("Preparing dataset...")
  local loader = Loader.new(model, cfg, 'train')
  local X,y = loader:getDataset()
  --X = torch.squeeze(X)
  print("Dataset prepared!")
  print("Training....")
  print(X:size())
  print(y:size())
  x, gradParams = model:getParameters()
  local criterion = backend.ClassNLLCriterion()
  criterion.sizeAverage = false
  local optimState = {learningRate = lr}
  local currentLoss
  local startTime = os.time()


  for i=1,y:size(1) do
    print(y[1])
  end
  -- training
  local currentLoss
  local startTime = os.time()
  for i=1, cfg.model.epochs do
    local averageLoss = 0
    for j = i*10, i*10+10 do
      -- local function we give to optim
      -- it takes current weights as input, and outputs the loss
      -- and the gradient of the loss with respect to the weights
      -- gradParams is calculated implicitly by calling 'backward',
      -- because the model's weight and bias gradient tensors
      -- are simply views onto gradParams
      feval = function(x)
         gradParams:zero()
         input = X[j]:clone()
         unsqueeze = nn.Unsqueeze(1)
         input = unsqueeze:forward(input)
         label = y[j]:clone()
         local output = model:forward(input)
         print(output:size())
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
    --torch.save('epoch.t7', o) --TODO
  end
end

local createModel = dofile 'createModel.lua'
local model = createModel(nn, 5, 1, 'train')
train(model, 0.001, true)
