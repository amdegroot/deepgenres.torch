require 'nn'


function loadModel(pkg, numClasses, imgSize, phase)
  backend = pkg
  batch = nil
  if phase == 'train' then
    batch = imgSize
  else
    batch = 1
  end


  -- Input of shape batch X imgSize X imgSize
  -- Torch automatically infers batch size
  model = backend.Sequential()
  model:add(backend.SpatialConvolution(batch,64,3,3,1,1,1,1))
  model:add(backend.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.SpatialConvolution(64,128,3,3,1,1,1,1))
  model:add(backend.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.SpatialConvolution(128,256,3,3,1,1,1,1))
  model:add(backend.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.SpatialConvolution(256,512,3,3,1,1,1,1))
  model:add(backend.ELU())
  model:add(backend.SpatialMaxPooling(2,2))
  model:add(backend.View(-1,512))
  model:add(backend.Linear(512,1024))
  model:add(backend.ELU())
  model:add(backend.Dropout(0.5))
  model:add(backend.View(512*8*8*2))
  model:add(backend.Linear(512*8*8*2,numClasses))
  model:add(backend.LogSoftMax())

  return model
end

return loadModel
