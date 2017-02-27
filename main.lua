
require 'torch'
require 'loader'
require 'songLoader'
require 'sliceSpectrogram'
local cfg = require 'config'

local cmd = torch.CmdLine()
cmd:option('-loadModel', false, 'Load previously saved model')
cmd:option('-mode', 'train', 'Training mode')
cmd:option('-modelName', 'DeepGenre', 'Name of class containing architecture')
cmd:option('-nGPU', -1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-epochSave', false, 'save model every epoch')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-saveFileName', 'deepgenre.t7', 'Name of model to save as')
cmd:option('-epochs', 20, 'Number of epochs for training')
cmd:option('-learningRate', 0.001, ' Training learning rate')
--cmd:option('-momentum', 0.90, 'Momentum for SGD')
cmd:option('-batchSize', 128, 'Batch size in training')
--cmd:option('-permuteBatch', false, 'Set to true if you want to permute batches AFTER the first epoch')
cmd:option('-validationBatchSize', 10, 'Batch size for validation')
cmd:option('-numHidden', 6, 'Number of hidden layers')


local opt = cmd:parse(arg)

if opt.nGPU == -1 then backend = nn else backend = cudnn end

local optimState = {
    lr = opt.learningRate
}

if opt.mode == 'train' then
  train_X, train_y = getDataset(cfg.files, cfg.genres, cfg.slice, cfg.valRatio, cfg.testRatio, mode="train")
  model = createModel(backend, numClasses, sliceSize, 'train')
  print("[+] Training the DeepGenre net...")
  train(model, params)
  print("Finished!")
end

function train(model, optimState.lr)
  print("Preparing dataset...")
  X,y = getDataset(cfg.files, cfg.genres, cfg.slice, cfg.model.valRatio, cfg.model.testRatio, 'train')
  print("Dataset prepared!")
  print("Training....")

  local criterion = backend.CrossEntropy()
