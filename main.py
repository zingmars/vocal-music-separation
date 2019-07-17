# Entry point for a script to attempt vocal and music separation using Tensorflow
# Accompanying thesis: Vocal and Audio separation using deep learning for Riga Technical University.
# Author: Ingmars Daniels Melkis <contact@zingmars.me>, Student ID: 161RDB280

import logging
import argparse
import os
import sys
from dataset import Dataset
from model import Model
from song import Song
from config import prepare_config
from evaluate import Evaluator

# Set up - Load config, arguments and set up logging
config = prepare_config('config.ini')
if config.get("logging", "logtype") == "file":
    logging.basicConfig(filename=config.get('logging', 'logfile'), level=logging.getLevelName(config.get('logging', 'loglevel')), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
elif config.get("logging", "logtype") == "console":
    logging.basicConfig(level=logging.getLevelName(config.get('logging', 'loglevel')), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

parser = argparse.ArgumentParser(description="Neural network for vocal and music splitting")
parser.add_argument("--mode", default="train", type=str, help="Mode in which the script is run (train/separate/evaluate).")
parser.add_argument("--weights", default="network.weights", type=str, help="File containing the weights to be used with the neural network. Will be created if it doesn't exist. Required for separation. Default is network.weights.")
parser.add_argument("--datadir", default="data", type=str, help="Directory in which the training data is located in. Default is data. (requires --mode=train)")
parser.add_argument("--validationdir", default="data-valid", type=str, help="Directory in which the validation data is located in. Default is data-valid. (requires --mode=train)")
parser.add_argument("--evaluationdir", default="evaluate", type=str, help="Directory in which separated data and the originals are located in. Default is evaluate. (requires --mode=evaluate)")
parser.add_argument("--epochs", default=1, type=int, help="How many times will the network go over the data. default - 1. (requires --mode=train)")
parser.add_argument("--file", default="mixture.wav", type=str, help="Name of the file from which to extract vocals. (requires --mode=separate)")
parser.add_argument("--output", default="vocals.wav", type=str, help="Name of the file to which the vocals will be written to. (requires --mode=separate)")
parser.add_argument("--dump_data", default="false", type=str, help="If set to true, dumps raw data for everything. Takes up a lot of space, but can be potentially useful for comparing results. (requires --mode=separate)")
parser.add_argument("--save_accompaniment", default="false", type=str, help="If set to true, the accompaniment will also be saved as a separate file (requires --mode=separate)")
args = parser.parse_args()

logging.log(55, 'Script started.')
if args.mode == "train":
    logging.info("Preparing to train a model...")
    dataset = Dataset(logging, config)
    dataset.load(args.datadir)
    dataset.get_data_for_cnn()
    dataset.get_labels_for_cnn()
    validation_set = Dataset(logging, config)
    validation_set.load(args.validationdir)
    validation_set.get_data_for_cnn()
    validation_set.get_labels_for_cnn()
    model = Model(logging, config, dataset, validation_set)
    model.build(output_summary=True)
    if os.path.isfile(args.weights):
        logging.info("Found existing weights, loading them...")
        model.load(args.weights)
    model.train(args.epochs, save_log=config.getboolean("model", "save_history"), log_name=config.get("model", "history_filename"))
    logging.info("Saving weights...")
    model.save(args.weights)
elif args.mode == "separate":
    logging.info("Preparing to separate vocals from instrumentals...")
    mixture = Song(logging, "a mixture", config)
    mixture.load_file(args.file)
    mixture.compute_stft(keep_spectrogram=True)
    dump_data = True if args.dump_data.lower() in ("yes", "true", "y", "t", "1") else False
    save_accompaniment = True if args.save_accompaniment.lower() in ("yes", "true", "y", "t", "1") else False
    if dump_data is True:
        mixture.dump_amplitude("original")
        mixture.dump_spectrogram("original")
    model = Model(logging, config)
    model.build()
    if os.path.isfile(args.weights):
        model.load(args.weights)
    else:
        logging.critical("Couldn't find a weights file.")
        sys.exit(11)
    if dump_data is True:
        model.isolate(mixture, args.output, save_accompaniment=save_accompaniment, save_original_mask=True, save_original_probabilities=True)
        mixture.dump_spectrogram("processed")
    else:
        model.isolate(mixture, args.output)
elif args.mode == "evaluate":
    logging.info("Preparing to evaluate the effectiveness of an output")
    evaluator = Evaluator(logging, config)
    evaluator.load_data(args.evaluationdir)
    evaluator.prepare_data()
    sdr, sir, sar = evaluator.calculate_metrics()
    evaluator.print_metrics(sdr, sir, sar)
else:
    logging.critical("Invalid action - %s", args.mode)
    sys.exit(12)
logging.log(56, "Script finished!")
