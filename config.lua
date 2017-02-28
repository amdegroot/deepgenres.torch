local cfg = {
  genres = {'classical','country','hip-hop','rock','tropical-house'},


  files = 250, -- num slices per genre to use in final dataset
  valRatio = 0.3, -- validation ratio
  testRatio = 0.1, -- test ratio
  slicesPerGenre = 1000, -- num slices per genre initially created
  slice = 128, -- slice size

  -- Directories for files
  dir = {
    data = '../Data/', -- original music directory separated by genre
    spec = "./Spectrograms/", -- store initial spectrograms here
    slices = "./Slices/", -- store ~2 second spectrogram slices here
    dataset = "./Dataset/", -- store serialized instances and labels for train and test
    raw = "./Raw/" -- store tmp files
  },

  -- Spectrogram params
  spect = {
    pps = 50, -- spectrogram resolution (pixels per second)
    sliceSz = 128, -- slice size
    stride = 128,
    window_type = 'rect'
  },

  -- Model Params
  model = {
    batchSize = 1, -- input size: 1 x slice x slice
    lr = 0.001,
    epochs = 20
  }
}

return cfg
