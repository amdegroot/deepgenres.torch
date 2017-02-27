require 'torch'
require 'paths'
require 'nn'
require 'utils'
require 'sliceSpectrogram'
local cfg = dofile "config.lua"
local utils = paths.dofile('utils.lua')
local Loader = torch.class('Loader')

local function getDatasetName(numFiles, sliceSize)
  local name = numFiles..'_'..sliceSize
  return name
end

--TODO add ability to save/load val/test sets
local function saveDataset(train_X, train_y)
  --Create path for dataset if not existing
  if not paths.dirp(cfg.dir.dataset) then
    status = pcall(function() return paths.mkdir(cfg.dir.dataset) end)
    if not status then
      print("Error, race condition when making directory: "..self.cfg.dir.dataset)
      return 0
    end
  end
    --SaveDataset
    print("[+] Saving dataset... ")
    name = getDatasetName(cfg.files, cfg.slice)
    torch.save(cfg.dir.dataset.."train_X_"..name..".t7", train_X)
    torch.save(cfg.dir.dataset.."train_y_"..name..".t7", train_y)
    print("    Dataset saved! ✅💾")
end

local function createDatasetFromSlices(genreSz, genres, sliceSz, slicePath, valRatio, testRatio)
  local num = 250*#genres --#genres*genreSz
  local data = torch.Tensor(num,sliceSz,sliceSz)
  -- local labels = torch.Tensor(num,#genres):zero()
  local labels = torch.Tensor(num):zero()
  idx = 1
  for label,genre in pairs(genres) do
      print("-> Adding ...", genre)
      --Get slices in genre subfolder
      filenames = utils.slice(paths.files(slicePath..genre, '.png'),1,genreSz,1)

      -- Randomize file selection for this genre
      --TODO: shuffle(filenames)

      --Add data (X,y)
      for i,fn in pairs(filenames) do
        --print('hi')
        -- print(i)
        -- print(label)
        if i > 250 then break end -- only want the first genreSz songs
        img = utils.getImageData(slicePath..genre.."/"..fn, sliceSz)
        data[idx] = img
        -- labels[idx][label]= 1
        labels[idx] = label
        idx = idx + 1
      end
    end
    --TODO:Shuffle data
    -- Set sizes
    -- indexes = torch.Tensor({2,1,3}):long()
    -- input = torch.rand(5,5)
    -- selected = input:index(1,indexes)

    val_ = utils.toInt(num*valRatio)
    test_ = utils.toInt(num*testRatio)
    train_ = num-(val_+test_)
    shuffle = torch.randperm(train_):long()
    -- input = trainData.data[shuffle[i]]
    -- Split train/val/test data
    -- local shuffle = torch.randperm(data:size(1))


        -- shuffledData[idx] = data[shuffle[i]]
        -- shuffledLabels[idx] = labels[shuffle[i]]
        -- idx = idx + 1

-- :index(1,shuffle)
-- :index(1,shuffle)
    train_X = data[{{1,train_},{},{}}]
    -- train_y = labels[{{1,train_},{}}]
    train_y = labels[{{1,train_}}]

    train_X = train_X:index(1,shuffle):contiguous()
    train_y = train_y:index(1,shuffle):contiguous()
    -- print(train_y:size())
    -- for j =1, train_y:size(1) do
    --   print(train_y[j])
    -- end
    print("Dataset created! ✅")
    --Save
    saveDataset(train_X,train_y)
    -- saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, genres, sliceSize)
    return train_X,train_y
end


function Loader:__init(model, cfg, phase)
  -- bounding box data (defined in pixels on original image)
  self.model = model
  self.cfg = cfg
  self.phase = phase
  self.valRatio = cfg.valRatio
  self.testRatio = cfg.testRatio
  self.genres = cfg.genres
  local name = getDatasetName(cfg.files, cfg.slice)
  self.name = name
end

-- Create a dataset name with num samples per genre and slice size

--TODO add ability to save/load val/test sets
local function loadDataset(mode)
    -- Load existing data
    name = getDatasetName(cfg.files, cfg.slice)
    if mode == "train" then
        print("[+] Loading training and validation datasets... ")
        train_X = torch.load(cfg.dir.dataset.."train_X_"..name..".t7")
        train_y = torch.load(cfg.dir.dataset.."train_y_"..name..".t7")
        -- validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath,datasetName), "rb" ))
        -- validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath,datasetName), "rb" ))
        print("--> Training and validation datasets loaded! ✅")
        return train_X,train_y
    end
    -- else
    --   print("[+] Loading testing dataset... ")
    --   test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath,datasetName), "rb" ))
    --   test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath,datasetName), "rb" ))
    --   print("    Testing dataset loaded! ✅")
    --   return test_X, test_y
end





function Loader:getDataset()
  print("[+] Dataset name: "..self.name)
  if not paths.filep(self.cfg.dir.dataset.."train_X_"..getDatasetName(self.cfg.files, self.cfg.slice)..".t7") then
        print("[+] Creating dataset with ",self.cfg.files, " slices of size ", self.cfg.slice, " per genre... ⌛️")
        createDatasetFromSlices(self.cfg.files, self.genres, self.cfg.slice, self.cfg.dir.slices,self.valRatio, self.testRatio)
  else  print("[+] Using existing dataset") end
  return loadDataset(self.phase)
end
