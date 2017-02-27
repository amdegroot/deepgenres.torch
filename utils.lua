require 'audio'
require 'image'
--local cfg = dofile 'config'

local function getProcessedData(img, imageSize)
  local img = img:resize(1,imageSize,imageSize)
  img = img/255
  return img
end

local function getImageData(fn, imageSize)
  local img = image.load(fn)
  local imgData = getProcessedData(img, imageSize)
  return imgData
end

local function split(input, sep)
  if sep == nil then sep = "%s" end
  local t={} ; i=1
  for str in string.gmatch(input, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

local function slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl()
  end

  return sliced
end

local function getLabelTable(genre)
  label = {}
  for k,v in pairs(cfg.genres) do
    if v == genre then label[k] = 1 else label[k] = 0 end
  end
  return label
end

local function getLabel(genre)
  for k,v in pairs(cfg.genres) do
    if v == genre then return k end
  end
end

local function toInt(val)
  return math.floor(val + 0.5)
end

local function isMono(songFile)
  local song = audio.load(songFile)
  return song:size(2)==1
end

local function splitString(str, delimiter)
    result = {};
    for match in (str..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result;
end

local function contains(table, val)
   for i=1,#table do
      if table[i] == val then return true end
   end
   return false
end

local utils = {}

utils.getProcessedData = getProcessedData
utils.getImageData = getImageData
utils.split = split
utils.slice = slice
utils.getLabelTable = getLabelTable
utils.getLabel = getLabel
utils.toInt = toInt
utils.isMono = isMono
utils.contains = contains
utils.splitString = splitString
return utils
