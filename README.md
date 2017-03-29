# deepgenres.torch
A deep learning method for automatically labeling songs by genre using Torch.  The primary reason for creating this was to become more familiar with the audio field within deep learning.    

## Architecture
<img align="center" src = "https://github.com/amdegroot/deepgenres.torch/blob/master/doc/network.png" height = 200/>

## Pre-Processing
 1. First, because music from SoundCloud is free, we take advantage of a pretty sweet SoundCloud song scraper called [SoundScrape](https://github.com/Miserlou/SoundScrape), which you can install with `pip install soundscrape`. 
 To give you an example of how to install songs by genre:
      1. Simply type the genre into the SoundCloud search bar online, select Playlists on the sidebar and pick a set. 
      2.  Navigate to one level outside this repo and open up a terminal to create a Data folder (we assume you download all sets there).
      ```Shell
      mkdir Data
      cd Data
      ```
      3.  Then Open up a terminal and download each set with SoundScrape:
      ```Shell
      soundscrape https://soundcloud.com/full-url-to-the-selected-genre-set
      ```
      4.  You should create a separate folder containing the tracks for each genre inside the Data folder. 
      5.  Repeat this process for each genre of your choice and then update config.lua to support the genres you've selected.
          * Currently the config file is set up for Classical, Country, Hip-Hop, Rock, and (of course) Tropical-House.
        
 
 2. Next, using sox, we combine the two stereo channels to mono, and convert all of our songs to spectrograms so that they can be processed by the Conv-Net in a way that is similar to normal images.
 <img align="center" src= "https://github.com/amdegroot/deepgenres.torch/blob/master/doc/spectrogram_example.png"/>
 
 3. In order to make the most of the data we have, we slice-up these spectrograms to produce several small ~2 second clips that we can train on and treat as individual instances.
 <img align="center" src= "https://github.com/amdegroot/deepgenres.torch/blob/master/doc/sliced_spec_example.png"/>
 
## Full Pipeline
Note: Conversion to mono, conversion to spectrogram, and spectrogram slicing is all done by default when running `th train.lua`.  This can be changed for subsequent runs by changing the flag createSpectrograms to false.
 <img align="center" src= "https://github.com/amdegroot/deepgenres.torch/blob/master/doc/pipeline.png"/>
 
## Usage
To tweak the configuration to your setup, you may have to change some of the opts inside either of the following:
### Training 
run  `train.lua` 
### Testing 
To test the accuracy of a trained model run `test.lua`.  

### Results
Training on Soundcloud sets containing the genres metioned above (Tropical-House etc.) we were able to achieve an overall 
classification accuracy of 95%.  
Keep in mind the model is able to make these predictions from just ~2 second audio clips, so 
that's pretty cool.
 
## References
All of the above images can be found on Julien Despois' [blog post](https://chatbotslife.com/finding-the-genre-of-a-song-with-deep-learning-da8f59a61194).  This code is largely based off of his original TensorFlow implementation which can be found [here](https://github.com/despoisj/DeepAudioClassification).
