import os
import sys
import logging
from song import Song
import numpy as np

# Dataset: Loads and passes test data to the model
class Dataset:
    def __init__(self, logger, config):
        self.logger=logger
        self.config=config
        # Raw data
        self.mixtures = []
        self.vocals = []
        # Outputs for CNN
        self.mixture_windows = []
        self.labels = []

    # Load mixture and vocals and generates STFT for them
    def load(self, folder):
        if os.path.isdir(folder):
            for root, dirs, files in os.walk(folder):
                for file in filter(lambda f: f.endswith(".wav"), files):
                    self.logger.info("Loading song %s and computing stft for it.", os.path.join(root, file))
                    song_type = os.path.splitext(file)[0].lower()
                    if song_type == "mixture" or song_type == "vocals":
                        song = Song(self.logger, os.path.basename(root), self.config)
                        song.load_file(os.path.join(root,file))
                        song.compute_stft()
                        if(song_type == "mixture"):
                            self.mixtures.append(song)
                        elif(song_type == "vocals"):
                            self.vocals.append(song)
                        self.logger.debug("%s loaded successfully.", song_type)
                    else:
                        self.logger.debug("File %s is not named correctly. Ignoring...", song_type)
        else:
            self.logger.critical("Folder %s does not exist!", folder)
            sys.exit(8)
        if(len(self.mixture) != len(self.vocals)):
            self.logger.critical("There doesn't appear to be a vocal track for each mixture (or the other way around).")
            sys.exit(15)

    def get_data_for_cnn(self):
        length = self.config.getint("song", "sample_length")
        self.logger.info("Preparing data of type 'mixture' for the CNN...")
        if len(self.mixtures) == 0:
            self.logger.critical("No mixtures for training found. Did you name them wrong?")
            sys.exit(9)
        self.logger.debug("Preparing %i songs...", len(self.mixtures))
        amplitudes = None
        for num in range(0, len(self.mixtures)):
            if amplitudes is None:
                amplitudes = self.mixtures[0].split_spectrogram(length)
            else:
                amplitudes = np.vstack((amplitudes, self.mixtures[0].split_spectrogram(length)))
            del self.mixtures[0]
        self.logger.debug("Got %i slices. Each slice has %i frequency bins, and each frequency bin has %i time slices.", len(amplitudes), len(amplitudes[0]), len(amplitudes[0][0]))
        self.logger.debug("Adding a 4th dimension to placate the CNN model...")
        # Add a dimension to make the CNN accept the data. Signifies that we have a greyscale "picture"
        amplitudes = np.array(amplitudes).reshape(len(amplitudes), len(amplitudes[0]), len(amplitudes[0][0]), 1)
        self.mixture_windows = amplitudes

    def get_labels_for_cnn(self):
        length = self.config.getint("song", "sample_length")
        self.logger.info("Preparing data of type 'vocals' for the CNN...")
        if len(self.vocals) == 0:
            self.logger.critical("No original vocals for training found. Did you name them wrong?")
            sys.exit(10)
        self.logger.debug("Preparing %i songs...", len(self.vocals))
        labels = []
        for num in range(0, len(self.vocals)):
            labels.extend(self.vocals[0].get_labels(length))
            del self.vocals[0]
        self.logger.debug("Got %i slices.", len(labels))
        self.labels = np.array(labels)
