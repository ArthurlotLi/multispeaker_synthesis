#
# speaker_encoder_test.py
#
# Testing harness for speaker encoder models. Allows for the chain
# testing of multiple model variants at once with the test set. 
# 
# By default expects the data to be in ./datasets/SV2TTS/encoder_test.
#
# Usage:
# python speaker_encoder_test.py ./production_models/speaker_encoder/model1
# python speaker_encoder_test.py ./saved_models/encoder_model4
# python speaker_encoder_test.py ./saved_models/encoder_model4 -d ./datasets/SV2TTS/encoder_test_original


from utils.argutils import print_args
from speaker_encoder.test import batch_test

from pathlib import Path
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Tests the speaker encoder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument("model_batch_dir", type=str, help= \
    "Location for all of the models to batch test.")
  parser.add_argument("-d","--clean_data_root", type=Path, default="./datasets/SV2TTS/encoder_test", help= \
    "Path to the output directory of encoder_preprocess.py.")
  parser.add_argument("-t", "--test_report_dir", type=Path, default="./evaluation_results/speaker_encoder", help=\
    "Path to where the test report should be placed.")
  parser.add_argument("-c", "--use_cpu", action="store_true", help= \
    "Force usage of the CPU - will try to use GPU otherwise.")
  parser.add_argument("-p", "--projection", action="store_true", help= \
    "Generate a projection visualizing model embedding for each model.")
  args = parser.parse_args()

  # Run the training
  print_args(args, parser)
  batch_test(**vars(args))