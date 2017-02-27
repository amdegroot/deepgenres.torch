local cfg = {
  genres = {'classical','country','hip-hop','rock','tropical-house'},


  files = 250, -- each of these is a small slice
  valRatio = 0.3,
  testRatio = 0.1,

  slice = 128, -- slice size
  -- Paths for files
  dir = {
    data = '../Data/',
    spec = "./Spectrograms/",
    slices = "./Slices/",
    dataset = "./Dataset/",
    raw = "./Raw/"
  },
  spect = {
    pps = 50, -- spectrogram resolution (pixels per second)
    sliceSz = 128, -- slice size
    --window_size = 50,
    stride = 128,
    window_type = 'rect'
  },

  -- Model Params
  model = {
    batchSize = 128,
    lr = 0.001,
    epochs = 20
  }

}

return cfg
