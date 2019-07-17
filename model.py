import os
import sys
import numpy as np
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from dataset import Dataset

# Class to manage the model, it's state.
class Model:
    def __init__(self, logger, config, dataset=None, validation_data=None):
        self.logger = logger
        self.config = config
        self.model = None
        self.dataset = dataset
        self.validation_data = validation_data

    # Build a model
    # If you want to experiment, this is the place you want to make the changes in
    def build(self, output_summary=False):
        self.logger.info("Building the model...")
        model = Sequential()

        bins = math.ceil(self.config.getint("song", "window_size")/2)+1
        model.add(Conv2D(32, (3,3), padding="same", input_shape=(bins, self.config.getint("song", "sample_length"), 1), activation='relu'))
        model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
        model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('elu'))
        model.add(Dropout(0.5))
        model.add(Dense(bins, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        if output_summary is True:
            model.summary()
        self.model = model

    def train(self, epochs, batch=32, save_log=False, log_name="history.csv"):
        if self.model is not None:
            self.logger.info("Training the model...")
            self.logger.info("Beggining training with %i samples.", len(self.dataset.mixture_windows))
            weights_backup = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=5)
            #TODO: Evaluate the use of fit_generator or train_on_batch to load data from disk instead of storing it all in RAM since it takes up a lot of memory otherwise.
            training = self.model.fit(self.dataset.mixture_windows, self.dataset.labels, batch_size=batch, epochs=epochs, validation_data=(self.validation_data.mixture_windows, self.validation_data.labels), callbacks=[weights_backup])
            self.logger.info("Training finished.")
            if save_log is True:
                self.logger.info("Exporting statistics.")
                import csv
                with open(log_name, 'w') as f:
                    w = csv.DictWriter(f, training.history.keys())
                    w.writeheader()
                    w.writerow(training.history)

    def save(self, filename='obj.save'):
        if self.model is not None:
            self.model.save_weights(filename, overwrite=True)
        else:
            self.logger.critical("Cannot save weights - model not set up")
            sys.exit(5)

    def load(self, filename='obj.save'):
        if self.model is not None and os.path.isfile(filename):
            self.model.load_weights(filename)
        else:
            self.logger.critical("Cannot load weights - model not set up or file not found")
            sys.exit(3)

    def isolate(self, mixture, output="output.wav", save_accompaniment=True, save_original_mask=False, save_original_probabilities=False):
        if self.model is not None:
            #TODO: For some reason the output loses ~0.004s (3*BINS+1) worth of samples
            self.logger.info("Preparing the song...")
            split_x = np.array(mixture.split_slidingwindow(self.config.getint("song", "sample_length")))
            split_x = split_x.reshape(len(split_x), len(split_x[0]), len(split_x[0][0]), 1)
            self.logger.info("Extracting vocals from the audio file...")
            prediction = self.model.predict(split_x)
            prediction = np.transpose(prediction) # Transpose the mask into the format librosa uses
            if save_original_probabilities is True:
                np.savetxt('original_predicted_probabilities.out', prediction)
            self.logger.info("Calculating the binary mask...")
            # Probability to label conversion, as there's no other way to get the output from the network in the right format
            #TODO: don't fill the accompaniment array if save_accompaniment is set to False
            accompaniment = np.zeros(np.shape(prediction))
            for x in range(0, len(prediction)):
                for y in range(0, len(prediction[x])):
                    prediction[x][y] = 1 if prediction[x][y] > 0.45 else 0 # Higher values tend to make voice unintelligible.
                    accompaniment[x][y] = 0 if prediction[x][y] > 0.45 else 1
            if save_original_mask is True:
                np.savetxt('predicted_mask.out', prediction)
            if save_accompaniment is True:
                spectrogram_bak = mixture.get_spectrogram()
                mixture.apply_binary_mask(accompaniment)
                mixture.reverse_stft()
                mixture.save_file("instrumental_"+output)
                mixture.set_spectrogram(spectrogram_bak)
            mixture.apply_binary_mask(prediction)
            mixture.reverse_stft()
            mixture.save_file(output)
        else:
            self.logger.critical("Model not set up, cannot attempt to isolate.")
            sys.exit(4)
