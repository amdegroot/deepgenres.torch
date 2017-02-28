require 'torch'
require 'image'
require 'paths'
local utils = paths.dofile('utils.lua')
local cfg = paths.dofile('config.lua')
local limitPer = cfg.slicesPerGenre
local currentNum = 0
local genre

 -- Creates slices from spectrogram
local function sliceSpectrogram(fn, size)
	if currentNum > limitPer then -- skip tracks if already reached limit per genre
		local nextGenre = utils.split(paths.basename(fn),"_")[1]
		if not (nextGenre == genre) then
			genre = nextGenre
			currentNum = 0
			print("Moving to next genre...")
		end
		print("Skipping extra track...")
		goto next
	end
	-- Load the full spectrogram
	local status, img = pcall(function() return image.load(cfg.dir.spec..fn,1) end)
	if not status then goto next end
	-- Compute approximate number of size x size samples
	local width = img:size()[2]
	local height = img:size()[1]
	local numSamples = math.floor(width/size)-1

	-- Create directory to hold slices if there isn't one already
	local slicePath = cfg.dir.slices..genre
	if not paths.dir(slicePath) then
		status = pcall(function () return paths.mkdir(slicePath) end)
		if not status then print("Error making slice path for "..genre.." music slices") return 0 end
	end

	for i=1,numSamples do
		print('Creating slice: '..i..'/'..numSamples..' for '..fn)
		local start = (i*size)+1
		local status,slice = pcall(function() return image.crop(img,start,1,start+size,size+1)end)
		if not status then goto next end
		local ext = paths.extname(fn)
		local base_fn = paths.basename(fn, ext)
		local numExt = utils.splitString(base_fn,"_")[2]
		local num = utils.splitString(numExt,".png")[1]
		local slice_fn = slicePath.."/"..genre.."_"..num.."_"..i..".png"
		image.save(slice_fn, slice)  -- TODO add a check here
		currentNum = currentNum + 1
	end
	::next:: -- Go here if skipping track
end

-- Slices all spectrograms
local function createSlices(size)
	genre = cfg.genres[1]	-- first genre to add e.g. tropical-house
	for file in paths.iterfiles(cfg.dir.spec) do
		print("Slicing :"..file)
		sliceSpectrogram(file,size)
	end

end

local slice_track = {}
slice_track.createSlices = createSlices
return slice_track
