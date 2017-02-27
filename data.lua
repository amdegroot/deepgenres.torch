require 'torch'
require 'paths'
require 'os'
require 'image'
require 'audio'
local utils = paths.dofile('utils.lua')
local createSlices = paths.dofile('sliceTrack.lua').createSlices
local cfg = paths.dofile('config.lua')

sliceSz = 128 -- tweak this for different freq range

local currentPath = paths.thisfile()

local function createSpectrogram(dir, fn, new_fn)
  local p = paths.cwd()..'/'
  local infile = "'"..dir..fn.."'"
  local tmpfile = "'"..p.."Raw/"..new_fn..".wav'"
  local imgfile = "'"..p.."Spectrograms/"..new_fn..".png'"
  local pps = cfg.spect.pps
  --TODO: I think I can just pass the filename without the extra strings for this to work
  -- if utils.isMono(infile) then
		-- cmd = "sox "..infile.." "..tmpfile
	-- else
  if not paths.filep(tmpfile) then
		cmd = "sox "..infile.." -c 1 "..tmpfile
    local status = os.execute(cmd)
    if not status then
  		print('ERROR converting '..infile..' to .wav format')
    end
  end

  cmd2 = "sox "..tmpfile.." -n spectrogram -Y 200 -X "..pps.." -m -r -o "..imgfile
  -- status, song, sample_rate = pcall(function () return audio.load(fn) end)
  local status2 = os.execute(cmd2)
  -- execution failed, corrupted audio file?
  if not status2 then print(string.format("Invalid audio file '%s'", fn)) return 0 end
  os.execute("rm "..tmpfile) -- remove the tmp .wav mono track
  return
end

local function toRawFolder(curdir, fn, new_fn, method)
  local rel_fn = "'"..paths.concat(curdir,fn).."'"
  local rename = "'../Data/"..curdir.."/"..new_fn
  print(rel_fn.."       "..rename)
  local status1 = os.rename(rel_fn,rename)
  if not status1 then print("Error, couldn't rename the file ") end
  local dest = "'"..paths.concat(cfg.dir.raw,new_fn)
  print(dest)
  local status2 = sys.execute(method..rename.." "..dest)
  if not status2 then
    print('Error executing % file: %s to %s', method, fn, cfg.dir.raw)
    return 0
  end
end



local function labelGenres(dirname)
  for dir in paths.iterdirs(dirname) do
    local idx=1
    for file in paths.iterfiles(dir) do
        local fn = paths.concat(dirname,dir,file)
        local base = dir..'_'..idx..paths.extname(file)
        local new_fn = paths.concat(dirname,dir,base)
        os.rename(file, new_fn)
        idx = idx + 1
    end
  end
end


local function createSpectrogramsFromAudio(cfg)
    genres = {}
  	--Create path if not existing
  	if not paths.dirp(cfg.dir.spec) then
  		status = pcall(paths.mkdir(cfg.dir.spec))
      if not status then
          -- pcall failed, race condition?
          print(string.format("Error, could not create directory:'%s'", cfg.dir.spec))
          return 0
      end
    end
  	-- Rename files according to genre
    for dir in paths.iterdirs(cfg.dir.data) do
      print(dir)
      local idx = 1
      for file in paths.iterfiles(cfg.dir.data..dir..'/') do
        print(file)
        --if not utils.contains(genres,dir) then goto next end
  		  print (string.format("Creating spectrogram for file %s/%s in %s...", idx, cfg.files, dir))
  		  local fileID = genres[dir]
  		  local new_fn = dir.."_"..idx
        idx = idx+1
  		  createSpectrogram(cfg.dir.data..dir.."/",file,new_fn)
      end
      ::next::
    end
end

-- Whole pipeline .mp3 -> .png slices
local function sliceAudio(cfg)
  print('Generating spectrograms...')
  createSpectrogramsFromAudio(cfg)
  print('Spectrograms Finished!')
  print('Slicing audio...')
  createSlices(cfg.slice)
  print('Slices created!')
end

local data = {}
data.sliceAudio = sliceAudio
data.labelGenres = labelGenres
return data
