require 'torch'
require 'paths'
require 'nn'
require 'utils'
require 'slice_track'

local cfg = paths.dofile('config.lua')
local utils = paths.dofile('utils.lua')
local Loader = torch.class('Loader')

local function getDatasetName(numFiles, sliceSize)
  local name = numFiles..'_'..sliceSize
  return name
end

--TODO add ability to save/load val/test sets
local function saveDataset(train_X, train_y, test_X, test_y)
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
    local train = {train_X,train_y}
    local test = {test_X,test_y}
    torch.save(cfg.dir.dataset.."train_data_"..name..".t7", train)
    torch.save(cfg.dir.dataset.."test_data_"..name..".t7", test)
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

      --Add data (X,y)
      for i,fn in pairs(filenames) do
        if i > 250 then break end -- only want the first genreSz songs
        img = utils.getImageData(slicePath..genre.."/"..fn, sliceSz)
        data[idx] = img
        -- labels[idx][label]= 1
        labels[idx] = label
        idx = idx + 1
      end
    end
    -- Set sizes
    val_ = utils.toInt(num*valRatio)
    test_ = utils.toInt(num*testRatio)
    train_ = num-(val_+test_)
    --Shuffle data
    shuffle = torch.randperm(data:size(1)):long()
    shuffledData = data:index(1,shuffle):contiguous()
    shuffledLabels = labels:index(1,shuffle):contiguous()

    train_X = shuffledData[{{1,train_},{},{}}]:clone()
    train_y = shuffledLabels[{{1,train_}}]:clone()
    test_X = shuffledData[{{train_,train_+val_},{},{}}]:clone()
    test_y = shuffledLabels[{{train_,train_+val_}}]:clone()
    print("Dataset created! ✅")
    --Save
    saveDataset(train_X,train_y,test_X,test_y)
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

--TODO add ability to save/load val sets
local function loadDataset(mode)
    -- Load existing data
    name = getDatasetName(cfg.files, cfg.slice)
    if mode == "train" then
      print("[+] Loading training and validation datasets... ")
      train_data = torch.load(cfg.dir.dataset.."train_data_"..name..".t7")
      print("--> Training and validation datasets loaded! ✅")
      return train_data[1], train_data[2]
    elseif mode == 'test' then
      print("[+] Loading testing dataset... ")
      test_data = torch.load(cfg.dir.dataset.."test_data_"..name..".t7")
      print("--> Testing dataset loaded! ✅")
      return test_data[1], test_data[2]
    else
      print("Not implemented yet!")
      return
    end
end



function Loader:getDataset()
  print("[+] Dataset name: "..self.name)
  if not paths.filep(self.cfg.dir.dataset.."train_X_"..getDatasetName(self.cfg.files, self.cfg.slice)..".t7") then
        print("[+] Creating dataset with ",self.cfg.files, " slices of size ", self.cfg.slice, " per genre... ⌛️")
        createDatasetFromSlices(self.cfg.files, self.genres, self.cfg.slice, self.cfg.dir.slices,self.valRatio, self.testRatio)
  else  print("[+] Using existing dataset") end
  return loadDataset(self.phase)
end
