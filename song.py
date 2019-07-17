import librosa
import numpy as np
import math
import os

# Song: Holds an information about a particular sound file and functions for modifying the raw sound data
# TODO: To decrease memory usage don't maintain all this data in RAM. Simply generate stfts, export them to a file and reuse them in the future.
class Song:
    def __init__(self, logger, name, config):
        self.logger=logger
        self.config=config
        self.name=name
        self.type=None
        self.amplitude=None
        self.spectrogram=None

    # Load a file and resample it if necessary. This is a good idea if the data is inconsistent or has more samples than we need (22kHz is more than enough)
    # Resampling is really slow though. You're better off resampling your audio manually beforehand.
    def load_file(self, filename):
        self.type=os.path.splitext(os.path.basename(filename))[0]
        self.logger.debug("Loading file %s of type %s", filename, self.type)
        self.data, _ = librosa.load(filename, sr=self.config.getint("song", "sample_size"), mono=True)
        self.logger.debug("File loaded.")

    def save_file(self, filename):
        #TODO: Don't save as a 32bit float since librosa can't load it afterwards
        self.logger.info("Saving audio data to %s", filename)
        librosa.output.write_wav(filename, self.data, self.config.getint("song", "sample_size"), norm=True)

    def dump_amplitude(self, note=""):
        if self.amplitude is not None:
            np.savetxt(self.name + '-' + self.type + '-' + note + '-amplitude.out', self.amplitude)

    def dump_spectrogram(self, note=""):
        if self.spectrogram is not None:
            np.savetxt(self.name + '-' + self.type + '-' + note + '-spectrogram.out', self.spectrogram)

    def get_spectrogram(self):
        return self.spectrogram
    def set_spectrogram(self, spectrogram):
        self.spectrogram = spectrogram

    def get_name(self):
        return self.name
    def get_raw_data(self):
        return self.data

    # Computes the short-term fourier transform and generates amplitude of the signals that the network can train on
    def compute_stft(self, keep_spectrogram=False, keep_data=False):
        if self.data is not None:
            self.logger.debug("Generating sftf for %s", self.name)
            spectrogram = librosa.stft(self.data, self.config.getint("song", "window_size"), hop_length=self.config.getint("song", "hop_length"))
            self.amplitude = librosa.power_to_db(np.abs(spectrogram)**2)
            if keep_spectrogram is True:
                self.spectrogram = spectrogram
            self.data = None
        else:
            self.logger.critical("Tried to generate stft for %s when the file wasn't loaded.", self.name)
            sys.exit(6)

    # Split the spectrogram into smaller blocks so that our network can work with it
    def split_spectrogram(self, length=25):
        # Each frequency bin (defined by window size) contains data for that particular
        # frequency over time - [frequency][time].
        # To make the data more paletable for the neural network, we need to split it.
        # Each time entry corresponds to hop_size/sample_rate (i.e. 5ms @ 44100 Hz with hop size 256)
        # We only need to predict the middle bin, the rest are there for context
        # FIXME: This loses a few ms of data from the input audio since it rounds down.
        # TODO: Save the spectrogram data to a file to avoid needing to generate and store it in RAM it every time
        slices = []
        for x in range (0, self.amplitude.shape[1] // length):
            _slice = self.amplitude[:,x * length : (x + 1) * length]
            slices.append(_slice)
        return slices

    def split_slidingwindow(self, length=25):
        # Similar to the previous function but we create a sliding window for each
        # bin. We do this only when predicting for a real song because of the memory requirements.
        height = self.amplitude.shape[0]
        # Pad the dataset
        amplitude = self.amplitude
        amplitude = np.column_stack((np.zeros((height, math.floor(length/2))), amplitude))
        amplitude = np.column_stack((amplitude, np.zeros((height, math.floor(length/2)))))
        slices = []
        for x in range(math.floor(length/2), amplitude.shape[1] - math.floor(length/2)):
            length_before = x - math.floor(length/2)
            length_after = x + math.floor(length/2)
            slices.append(np.array(amplitude[:, length_before : (length_after + 1)]))
        return slices

    def get_labels(self, length=25):
        # The labels contain the value of the middle slice of each time container
        # in each frequency. The network understands which slice to target eventually.
        # TODO: Maybe generate binary masks with librosa's softmask instead?
        slices = []
        for x in range(0, self.amplitude.shape[1] // length):
            _slice = []
            for y in range(0, self.amplitude.shape[0]):
                # Mark whether there's voice acitivity in the given freq. in the given time bin
                # The vocals might actually have some activity that equal silence but isn't a 0 in the original mix
                # so we have to filter these out accordingly.
                # NOTE: This _might_ clean out whispering and such. Test with your data set.
                if self.amplitude[y,x*length+(math.ceil(length/2) if length > 1 else 0)] > 1:
                    _slice.append(1)
                else:
                    _slice.append(0)
            slices.append(_slice)
        return slices

    # Apply network predictions and get useable output
    def apply_binary_mask(self, mask):
        self.spectrogram = np.multiply(self.spectrogram, mask)

    def reverse_stft(self):
        if self.amplitude is not None:
            self.data = librosa.istft(self.spectrogram, self.config.getint("song", "hop_length"), self.config.getint("song", "window_size"))
        else:
            self.logger.critical("Cannot find a STFT spectrogram to reverse - was it not generated?")
            sys.exit(7)
