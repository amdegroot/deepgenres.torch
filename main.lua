require 'torch'
require 'loader'
require 'songLoader'
require 'sliceSpectrogram'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()

cmd:option('-mode', 'test', 'Test mode')
cmd:option('-pretrained', true, 'Load pretrained weights')
cmd:option('-modelName', './models/epoch20.t7', 'Path to pre-trained model to load')
cmd:option('-backend', 'nn', 'Set to cudnn to use GPU')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-epochSave', false, 'save model every epoch')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-saveFileName', 'deepgenre.t7', 'Name of model to save as')
cmd:option('-epochs', 20, 'Number of epochs for training')
cmd:option('-learningRate', 0.001, ' Training learning rate')
cmd:option('-batchSize', 128, 'Batch size in training')
cmd:option('-numHidden', 6, 'Number of hidden layers')
cmd:text()

local opt = cmd:parse(arg)
print(opt)
local cfg = paths.dofile('config.lua')

local optimState = {
    lr = opt.learningRate
}

if opt.mode == "train" then
  local createModel = dofile 'createModel.lua'
  local model = createModel(nn, 5, 1, 'train')
  print("[+] Training the DeepGenre net...")
  train(model, 0.001, false)
  print("Finished!")

else
  test()

end


function test()
  local model = torch.load("./models/epoch20.t7")
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

function train(model, lr, createSpectrograms)
  -- if rename then rename(cfg.dir.data) end
  backend = nn
  if createSpectrograms then sliceAudio(cfg) end

  print("Preparing dataset...")
  local loader = Loader.new(model, cfg, "train")
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
  for i=1, cfg.model.epochs do
    local averageLoss = 0
    for j = 1, X:size(1) do
      -- local function we give to optim
      -- it takes current weights as input, and outputs the loss
      -- and the gradient of the loss with respect to the weights
      -- gradParams is calculated implicitly by calling 'backward',
      -- because the model's weight and bias gradient tensors
      -- are simply views onto gradParams
      feval = function(x)
         gradParams:zero()
         input = X[j]:clone()
         label = y[j]
         unsqueeze = backend.Unsqueeze(1) -- document this
         input = unsqueeze:forward(input)
         local output = model:forward(input)
         print(output:size())
         print(label)
         local loss = criterion:forward(output, label)
-----------------------------------------------------------------------------------

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
    --torch.save('epoch'..i..'.t7', model) -- save weights after each epoch
  end
  torch.save("deepgenre.t7", model)
end
