# A tool to obtain instrumentals from a mixture using a vocal track
# This is used to create files that can be compared with CNNs
# output. Because of this it also generates and uses binary masks
# so that any problems caused by that method don't impact the comparison.
import numpy as np
import librosa
import sys
import argparse

parser = argparse.ArgumentParser(description="A small tool to obtain instrumentals from a mixture provided a vocal track")
parser.add_argument("mixture", default="mixture.wav", type=str, help="Path to the mixture")
parser.add_argument("vocals", default="vocals.wav", type=str, help="Path to the vocal track")
args = parser.parse_args()

print("Loading files...")
mixture, mixture_sample_rate = librosa.load(args.mixture)
vocals, vocal_sample_rate = librosa.load(args.vocals)

if mixture_sample_rate != vocal_sample_rate:
    print("Sample rates don't match. Resample your data and try again.")
    sys.exit(2)

print("Processing audio data...")
mixture_spectrogram = librosa.stft(mixture, 2048, 256)
vocal_spectrogram = librosa.stft(vocals, 2048, 256)

vocal_amplitude = librosa.power_to_db(np.abs(vocal_spectrogram)**2)
mixture_magnitude, mixture_phase = librosa.core.magphase(mixture_spectrogram)

# Create a binary mask for the vocals
# We can actually work with just the raw spectrograms if we need to,
# but that's not the point of this script
print("Generating masks...")
vocal_mask = []
instrumental_mask = []
for x in range(0, vocal_amplitude.shape[0]):
    vocal_slice = []
    instrumental_slice = []
    for y in range(0, vocal_amplitude.shape[1]):
        # This is the same algorithm as used in song.py
        # Please refer to that file for more information on what it does
        # And the problems this has.
        if vocal_amplitude[x][y] > 0.01:
            vocal_slice.append(1)
            instrumental_slice.append(0)
        else:
            vocal_slice.append(0)
            instrumental_slice.append(1)
    vocal_mask.append(vocal_slice)
    instrumental_mask.append(instrumental_slice)

print("Applying masks...")
output_vocals = np.multiply(mixture_magnitude, vocal_mask) * mixture_phase
output_instrumentals = np.multiply(mixture_magnitude, instrumental_mask) * mixture_phase

print("Processing audio data...")
vocal_data = librosa.istft(output_vocals, 256, 2048)
instrumental_data = librosa.istft(output_instrumentals, 256, 2048)

print("Outputting files...")
librosa.output.write_wav("processed_vocals.wav", vocal_data, mixture_sample_rate, norm=True)
librosa.output.write_wav("processed_instrumentals.wav", instrumental_data, mixture_sample_rate, norm=True)
print("Completed. Output can be found in processed_vocals.wav and processed_instrumentals.wav.")
