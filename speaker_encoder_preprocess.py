#
# speaker_encoder_preprocess.py
#
# Harness for executing preprocessing on all speaker_encoder data.
# The speaker encoder is trained on a speaker verification task
# and uses three of our four datasets (VoxCeleb1, VoxCeleb2, and
# LibriSpeech).
# 
# This should only be run once to generate numpy arrays saved
# in the appropriate directories for the model training to 
# use. The products should be created with a designated split
# of test/dev/train, roughly at 10/10/80. The split should be
# based on the originally suggested splits of the component
# datasets (created to separate speakers).
#
# A word to the wise: when sourcing VoxCeleb2, Convert all m4a files
# into wav. librosa doesn't support m4a and uses audioread instead
# which is very slow. You may use the m4a_to_wav.py utility in 
# the utils subdirectory. This may the difference between a few
# hours to preprocess VoxCeleb2 and 21 days. No joke. 
#
# Usage (Train): 
# python speaker_encoder_preprocess.py ./datasets 
#
# Usage (Test):
# python speaker_encoder_preprocess.py ./datasets -d librispeech_test,voxceleb1_test,voxceleb2_test -o ./datasets/SV2TTS/encoder_test/ -s
#
# Additional options:
#  -o = Path to output directory - defaults to <datasets_root>/SV2TTS/encoder
#  -d = List of datasets to preprocess (only train sets will be used) - defaults to librispeech_other,voxceleb1,voxceleb2.
#  -s = Whether to skip existing output files. 

from speaker_encoder.preprocess import preprocess_librispeech, preprocess_test_librispeech, preprocess_test_voxceleb1, preprocess_voxceleb1, preprocess_voxceleb2
from utils.argutils import print_args

from pathlib import Path
import argparse

if __name__ == "__main__":
  class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

  parser = argparse.ArgumentParser(
    description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                "writes them to the disk. This will allow you to train the encoder. The "
                "datasets required are at least one of VoxCeleb1, VoxCeleb2 and LibriSpeech. "
                "Ideally, you should have all three. You should extract them as they are "
                "after having downloaded them and put them in a same directory in this layout:\n"
                "-[datasets_root]\n"
                "  -LibriSpeech\n"
                "    -train-other-500\n"
                "  -VoxCeleb1\n"
                "    -train\n"
                "      -wav\n"
                "    -vox1_meta.csv\n"
                "  -VoxCeleb2\n"
                "    -train\n"
                "      -aac",
    formatter_class=MyFormatter
  )

  # Grab the arguments necessary. 
  parser.add_argument("datasets_root", type=Path, help=\
    "Path to the directory containing your LibriSpeech/TTS and VoxCeleb datasets.")
  parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
    "Path to the output directory that will contain the mel spectrograms. If left out, "
    "defaults to <datasets_root>/SV2TTS/encoder/")
  parser.add_argument("-d", "--datasets", type=str,
                      default="librispeech_other,voxceleb1,voxceleb2", help=\
    "Comma-separated list of the name of the datasets you want to preprocess. Only the train "
    "set of these datasets will be used. Possible names: librispeech_other, voxceleb1, "
    "voxceleb2.")
  parser.add_argument("-s", "--skip_existing", action="store_true", help=\
    "Whether to skip existing output files with the same name. Useful if this script was "
    "interrupted.")
  args = parser.parse_args()

  # Split the datasets into a list.
  args.datasets = args.datasets.split(",")

  # If no out_dir provided, default to <datasets_root>/SV2TTS/encoder
  if not hasattr(args, "out_dir"):
      args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder")
  assert args.datasets_root.exists()
  args.out_dir.mkdir(exist_ok=True, parents=True)

  print_args(args, parser)

  print("[INFO] Speaker Encoder preprocessing starting...")

  # Given the chosen datasets, get started.
  preprocess_func = {
    "librispeech_other": preprocess_librispeech,
    "voxceleb1": preprocess_voxceleb1,
    "voxceleb2": preprocess_voxceleb2,
    "librispeech_test": preprocess_test_librispeech,
    "voxceleb1_test": preprocess_test_voxceleb1,
    "voxceleb2_test": preprocess_test_voxceleb2,
    "libirspeech_dev": preprocess_dev_librispeech,
  }
  args = vars(args)
  for dataset in args.pop("datasets"):
    print("Preprocessing %s" % dataset)
    preprocess_func[dataset](**args)

  print("[INFO] Speaker Encoder preprocessing complete.")