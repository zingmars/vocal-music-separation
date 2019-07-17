# Evaluate the accuracy of the neural network by calculating SDR (distortion)
# SIR (interference from other sources) and SAR (artifacts)
import numpy as np
import museval
import os
import sys
from song import Song

class Evaluator:
    def __init__(self, logger, config):
        self.logger=logger
        self.config=config
        self.vocals=None
        self.accompaniments=None
        self.estimated_vocals=None
        self.estimated_accompaniments=None
        self.names=None

    def load_data(self, folder):
        self.vocals=[]
        self.accompaniments=[]
        self.estimated_vocals=[]
        self.estimated_accompaniments=[]
        if os.path.isdir(folder):
            for root, firs, files in os.walk(folder):
                for file in filter(lambda f: f.endswith(".wav"), files):
                    song_type = os.path.splitext(file)[0].lower()
                    self.logger.info("Loading song %s.", os.path.join(root, file))
                    if song_type == "vocals" or song_type == "accompaniment" or song_type == "estimated_vocals" or song_type == "estimated_accompaniment":
                        song = Song(self.logger, os.path.basename(root), self.config)
                        song.load_file(os.path.join(root,file))
                        if(song_type == "vocals"):
                            self.vocals.append(song)
                        elif(song_type == "accompaniment"):
                            self.accompaniments.append(song)
                        elif(song_type == "estimated_vocals"):
                            self.estimated_vocals.append(song)
                        elif(song_type == "estimated_accompaniment"):
                            self.estimated_accompaniments.append(song)
                        self.logger.debug("%s loaded successfully.", song_type)
                    else:
                        self.logger.debug("File %s is not named correctly. Ignoring...", song_type)
        else:
            self.logger.critical("Folder %s does not exist!", folder)
            sys.exit(13)
        if (len(self.vocals) != len(self.accompaniments)) or (len(self.accompaniments) != len(self.estimated_vocals)) or (len(self.estimated_vocals) != len(self.estimated_accompaniments)):
            self.logger.critical("Array size mismatch. Did you misname a file?")
            sys.exit(14)

    # Extracts data from the dataset and sets the correct dimensions
    def prepare_data(self):
        self.names = []
        for element in range(0, len(self.vocals)):
            self.logger.debug("Processing %s...", self.vocals[element].get_name())
            self.names.append(self.vocals[element].get_name())
            self.vocals[element] = np.expand_dims(self.vocals[element].get_raw_data(), 1)
            self.accompaniments[element] = np.expand_dims(self.accompaniments[element].get_raw_data(), 1)
            self.estimated_vocals[element] = np.expand_dims(self.estimated_vocals[element].get_raw_data(), 1)
            self.estimated_accompaniments[element] = np.expand_dims(self.estimated_accompaniments[element].get_raw_data(), 1)
        self.vocals = np.array(self.vocals)
        self.accompaniments = np.array(self.accompaniments)
        self.estimated_vocals = np.array(self.estimated_vocals)
        self.estimated_accompaniments = np.array(self.estimated_accompaniments)
        # Since the neural net outputs slightly less data than in the original, we will cut off the part that we can't compare
        # Simply padding WOULD be a better idea, but we can't assume that the last few miliseconds have nothing going on in them.
        for element in range(0, len(self.vocals)):
            if np.shape(self.vocals[element])[0] > np.shape(self.estimated_vocals[element])[0]:
                self.logger.debug("Reshaping arrays for %s...", self.names[element])
                difference = np.shape(self.vocals[element])[0] - np.shape(self.estimated_vocals[element])[0]
                self.vocals[element] = self.vocals[element,:-difference,:]
                self.accompaniments[element] = self.accompaniments[element,:-difference,:]

    def calculate_metrics(self):
        sdr = sir = sar = []
        for element in range(0, len(self.vocals)):
            original_data = np.stack((self.vocals[element], self.accompaniments[element]))
            estimated_data = np.stack((self.estimated_vocals[element], self.estimated_accompaniments[element]))
            museval.metrics.validate(original_data, estimated_data)
            self.logger.info("Calculating metrics for %s...", self.names[element])
            obtained_sdr, _, obtained_sir, obtained_sar, _ = museval.metrics.bss_eval(original_data, estimated_data, window=np.inf, hop=0)
            if element == 0:
                sdr = obtained_sdr
                sir = obtained_sir
                sar = obtained_sar
            else:
                sdr = np.column_stack((sdr, obtained_sdr))
                sir = np.column_stack((sir, obtained_sir))
                sar = np.column_stack((sar, obtained_sar))
        return sdr, sir, sar

    def print_metrics(self, sdr, sir, sar):
        self.logger.info("Printing results...")
        for element in range(0, len(self.names)):
            self.logger.info("Song name: %s", self.names[element])
            self.logger.info("Vocals: SDR: %.2f, SIR: %.2f, SAR: %.2f", sdr[0][element], sir[0][element], sar[0][element])
            self.logger.info("Accompaniments: SDR: %.2f, SIR: %.2f, SAR: %.2f", sdr[1][element], sir[1][element], sar[1][element])
