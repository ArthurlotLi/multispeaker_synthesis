#
# synthesizer_train.py
#
# Executes training of the synthesizer model. Allows user to 
# specify various aspects of the model training process. A
# prerequisite of running this process is pretrained speaker
# encoder model, as it will be used in a transfer learning
# configuration. 
#
# Usage:
# python synthesizer_train.py model1 

from pathlib import Path

from synthesizer.hparams import hparams
from synthesizer.train import train
from utils.argutils import print_args
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help= \
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
        "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
        "states and restart from scratch.")
    parser.add_argument("--syn_dir", type=Path, default="./datasets/SV2TTS/synthesizer", help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("-s", "--save_every", type=int, default=0, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=100, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    args = parser.parse_args()
    print_args(args, parser)

    args.hparams = hparams

    # Run the training
    train(**vars(args))