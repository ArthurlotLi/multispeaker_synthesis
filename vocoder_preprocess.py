#
# vocoder_preprocess.py
#
# Pretty straightforward - utilizes the synthesizer dataset and 
# generates ground-truth aligned spectograms for the vocoder to learn
# to reach. 
#
# Usage (train):
# python vocoder_preprocess.py ./datasets
# 
# Usage (test):
# python vocoder_preprocess.py ./datasets -i ./datasets/SV2TTS/synthesizer_test -o ./datasets/SV2TTS/vocoder_test

import argparse
import os
from pathlib import Path

from synthesizer.hparams import hparams
from synthesizer.synthesize import run_synthesis
from utils.argutils import print_args



if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Creates ground-truth aligned (GTA) spectrograms from the vocoder.",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your SV2TTS directory. If you specify both --in_dir and "
        "--out_dir, this argument won't be used.")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="production_models/synthesizer/model6/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-i", "--in_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the mel spectrograms, the wavs and the "
        "embeds. Defaults to  <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help= \
        "Path to the output vocoder directory that will contain the ground truth aligned mel "
        "spectrograms. Defaults to <datasets_root>/SV2TTS/vocoder/.")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    args = parser.parse_args()
    print_args(args, parser)

    if not hasattr(args, "in_dir"):
        args.in_dir = args.datasets_root / "SV2TTS" / "synthesizer"
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root / "SV2TTS" / "vocoder"

    if args.cpu:
        # Hide GPUs from Pytorch to force CPU processing
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    run_synthesis(args.in_dir, args.out_dir, args.syn_model_fpath, hparams)
