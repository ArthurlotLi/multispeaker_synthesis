#
# synthesizer_test.py
#
# Testing harness for synthesizer models. Allows for the chain
# testing of multiple model variants at once with the test set. 
# 
# By default expects the data to be in ./datasets/SV2TTS/synthesizer_test.
#
# Usage:
# python synthesizer_test.py ./saved_models/synthesizer_model1
# python synthesizer_test.py ./production_models/synthesizer/model1


from utils.argutils import print_args
from synthesizer.test import batch_test

from pathlib import Path
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Tests the synthesizer.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument("model_batch_dir", type=str, help= \
    "Location for all of the models to batch test.")
  parser.add_argument("-d","--clean_data_root", type=Path, default="./datasets/SV2TTS/synthesizer_test", help= \
    "Path to the output directory of synthesizer_preprocess.py.")
  parser.add_argument("-t", "--test_report_dir", type=Path, default="./evaluation_results/synthesizer", help=\
    "Path to where the test report should be placed.")
  parser.add_argument("-c", "--use_cpu", action="store_true", help= \
    "Force usage of the CPU - will try to use GPU otherwise.")
  args = parser.parse_args()

  # Run the training
  print_args(args, parser)
  batch_test(**vars(args))