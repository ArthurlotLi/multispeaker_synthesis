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
# VoxCeleb: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ 
# LibriSpeech: https://www.openslr.org/12/

class SpeakerEncoderPreprocess:
  pass

if __name__ == "__main__":
  pass