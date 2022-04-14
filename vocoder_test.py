#
# vocoder_test.py
#
# Testing harness for vocoder models. Allows for the chain
# testing of multiple model variants at once with the test set. 
# 
# By default expects the data to be in ./datasets/SV2TTS/vocoder_test.
#
# Usage:
# python vocoder_test.py ./saved_models/model6_vocoder
# python vocoder_test.py ./production_models/vocoder/model1
# python vocoder_test.py ./production_models/vocoder/model6 -d ./datasets/SV2TTS/vocoder_test


from utils.argutils import print_args
from vocoder.sv2tts_test import batch_test

from pathlib import Path
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Tests the vocoder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument("model_batch_dir", type=str, help= \
    "Location for all of the models to batch test.")
  parser.add_argument("-d","--clean_data_root", type=Path, default="./datasets/SV2TTS/synthesizer_test", help= \
    "Path to the output directory of vocoder_preprocess.py.")
  parser.add_argument("-t", "--test_report_dir", type=Path, default="./evaluation_results/vocoder", help=\
    "Path to where the test report should be placed.")
  parser.add_argument("-c", "--use_cpu", action="store_true", help= \
    "Force usage of the CPU - will try to use GPU otherwise.")
  args = parser.parse_args()

  # Run the training
  print_args(args, parser)
  batch_test(**vars(args))