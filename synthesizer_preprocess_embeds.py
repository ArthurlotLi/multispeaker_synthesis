#
# synthesizer_preprocess_embeds.py
#
# Harness for executing preprocessing on all synthesizer data.
# The synthesizer utilizes LibriSpeech-clean dataset to
# learn how to output mel spectograms. 
#
# This part of preprocessing runs the speaker encoder and
# generates embeddings for all of the preprocessed audio from
# LibriSpeech-clean. Preprocess_audio should be run first. 
# 
# This should only be run once to generate numpy arrays saved
# in the appropriate directories for the model training to 
# use. 
#
# LibriSpeech: https://www.openslr.org/12/
#
# By default, uses model1 located in production_models.
#
# Usage (train): 
# python synthesizer_preprocess_embeds.py
#
# Usage (test):
# python synthesizer_preprocess_embeds.py --synthesizer_root ./datasets/SV2TTS/synthesizer_test

from synthesizer.preprocess import create_embeddings
from utils.argutils import print_args

from pathlib import Path
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--synthesizer_root", type=Path, default="./datasets/SV2TTS/synthesizer", help=\
    "Path to the synthesizer training data that contains the audios and the train.txt file. "
    "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
  parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                      default="./production_models/speaker_encoder/model1/encoder.pt", help=\
    "Path your trained encoder model.")
  parser.add_argument("-n", "--n_processes", type=int, default=10, help= \
    "Number of parallel processes. An encoder is created for each, so you may need to lower "
    "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
  args = parser.parse_args()

  # Preprocess the dataset
  print_args(args, parser)
  create_embeddings(**vars(args))