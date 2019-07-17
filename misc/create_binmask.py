# A script to generate a binary mask of an audio file using librosa and matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import argparse

parser = argparse.ArgumentParser(description="Create spectrogram from a wav file")
parser.add_argument("file", type=str, help="Path to the file (wav)")
parser.add_argument("mel", type=str, nargs='?', default="false", help="Should this spectrogram be rendered as a mel spectrogram? (true/false)")
parser.add_argument("output", default="", nargs='?', type=str, help="Path to output file (png)")
args = parser.parse_args()
audio, sampleRate = librosa.load(args.file)
plt.figure(figsize=(10,4))

mel = True if args.mel.lower() in ("yes", "true", "y", "t", "1") else False
amplitude = None
if mel is True:
    amplitude = librosa.power_to_db(np.abs(librosa.feature.melspectrogram(audio, sr=sampleRate)))
else:
    amplitude = librosa.power_to_db(np.abs(librosa.stft(audio)))

slices = []
for x in range(0, amplitude.shape[0]):
    _slice = []
    for y in range(0, amplitude.shape[1]):
        if amplitude[x][y] > 0.01:
            _slice.append(1)
        else:
            _slice.append(0)
    slices.append(_slice)

plt.imshow(slices, interpolation='nearest', cmap="Greys", origin='lower')
#librosa.display.specshow(slices, y_axis='mel', fmax=sampleRate, x_axis='time')
plt.title('Binary mask')
plt.tight_layout()
if args.output is not "":
    plt.savefig(args.output)
else:
    plt.show()
