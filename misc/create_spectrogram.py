# A script to generate a spectrogram of an audio file using librosa and matplotlib
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
spectrogram = None
if mel is True:
    spectrogram = librosa.feature.melspectrogram(audio, sr=sampleRate)
else:
    spectrogram = librosa.stft(audio)
librosa.display.specshow(librosa.power_to_db(np.abs(spectrogram), ref=np.max), y_axis='mel', fmax=sampleRate, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
if args.output is not "":
    plt.savefig(args.output)
else:
    plt.show()
