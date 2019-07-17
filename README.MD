Vocal and music seperation using a CNN
===

# Description

This CNN attempts to separate the vocals from the music. It does so by training on the amplitude data of the audio file and tries to estimate where the voiced parts are. Vocal separation is done by generating a binary mask of the time-frequency bins that the network thinks contain the vocals and applying it to the original file.

# Requirements

* Python 3
* Tensorflow (Tested with tensorflow-gpu), Keras
* And a few other python libraries that you can install by running `pip install -r requirements.txt` in the root directory

# Dataset

* The script was only tested with .wav files (16-bit and 24-bit signed wavs should work, 32-bit float doesn't). Other formats might work if your version of librosa is capable of opening it.
* Training data folder should have individual folder for each song. Each song should have two files - `mixture.wav` (the full song) and `vocals.wav` (original vocals). See below for a list of data sets that you could potentially use to train this network.
* To see an example of how the directory structures should look like, refer to [structure.md](structure.md).
* To make things faster, all songs should have the same sampling rate as configured (I only tested 22050kHz, but other sample rates should work) and should be in mono (if it isn't, the script will convert them, but the result isn't saved anywhere and it takes a while).

## Example data sets

* DSD100: https://sigsep.github.io/datasets/dsd100.html
* MUSDB: https://sigsep.github.io/datasets/musdb.html
* MedleyDB: https://medleydb.weebly.com/

# Setting up

1. pip install -r requirements.txt
2. py main.py

# Running

1. `python main.py -h` to see all arguments
2. `python main.py` will train the network with the default options
3. `python main.py --mode=separate --file=audio.wav` will attempt source separation on `audio.wav` and will output `vocals.wav`
4. `python main.py --mode=evaluate` will evaluate the effectiveness of audio source separation. More information below.

## Configuring

All relevant settings are located in the `config.ini` file. The file doesn't exist in the repository and will be automatically created and prepopulated with the default values on first run. For information on what each option does see `config.py`.

## Evaluating

This program also includes a simple wrapper around BSS-Eval which can be used to determine how effective audio source separation is. To use it you need - the original vocals (`vocals.wav`), the original accompaniment (`accompaniment.wav`), estimated vocals (`estimated-vocals.wav`) and estimated accompaniment (`estimated-accompaniment.wav`). If you don't have the accompaniment but have a mixture and vocals, you can use the `apply_vocal_mask.py` script in the `misc` folder. To get estimated accompaniment, you need to perform separation with the `--save_accompaniment` flag set to true. After you have all the files, create a data directory that contains a directory with the name of the song and copy all 4 files to it.

Note that librosa by default outputs a 32bit wav file which it can't load without ffmpeg, so you either need to add an extra conversion step between separating and evaluation or install ffmpeg and add it to your PATH. Both files need to have the same format and bitrate for evaluation to be successful as well.

### Bug

For a reason I haven't had the time to determine yet the neural network output sometimes has slightly less samples (~76 samples to be exact which is around 0.004s worth of samples) than the original. The evaluation script will account for this, but be advised that some samples are being lost during evaluation.

## Weights files when training

While training the network will save its weights every 5 epochs to avoid data loss should you have a power failure or a similar issue. These files may be deleted after training.

## Misc

The misc directory contains a few scripts that might be useful but aren't required to run the neural net.
