#
# production_embedding
#
# Allows for emebdding of speakers. Separate from produciton inference
# in the case that this is done offline first.
#
# Direct usage:
# python production_embedding.py ../multispeaker_synthesis_speakers

import speaker_encoder.inference as encoder
from pathlib import Path
import numpy as np
import argparse
from typing import Optional
import os

class Embedding:
  # Expects fpaths for the encoder and synthesizer. 
  def __init__(self, speaker_encoder_fpath, 
               verbose=True, load_immediately=True):
    self.speaker_encoder_fpath = speaker_encoder_fpath
    self.verbose = verbose
    
    if load_immediately:
      encoder.load_model(self.speaker_encoder_fpath)
  
  # Generates and saves an embedding given the location of wav file(s) 
  # to load. Expects the fpath to include the file name but not the
  # suffix. Ex:
  # "../multispeaker_synthesis_speakers/me/neutral" -> neutral.npy
  #
  # Alternatively, no filepath can be provided, and the embedding will
  # simply be returned. 
  def single_embedding(self, wav_fpath: Path, embed_fpath: Optional[Path] = None):
    if not encoder.is_loaded():
      encoder.load_model(self.speaker_encoder_fpath)

    # Compute the speaker embedding. 
    wav = encoder.preprocess_audio(wav_fpath)
    embed = encoder.embed_utterance(wav)
    if embed_fpath is not None:
      np.save(embed_fpath, embed, allow_pickle = False)
    else:
      return embed

  def directory_embedding(self, dir_path: Path, embed_fpath: Optional[Path] = None):
    """
    (Yes, I'm finally using the proper function comment style)

    Generates a single embedding using ALL wav files in the directory
    provided. Essentially, a speaker embedding, not utterance embedding.
    """
    if not encoder.is_loaded():
      encoder.load_model(self.speaker_encoder_fpath)
    
    dir_files = os.listdir(str(dir_path))

    # Compute the speaker embedding. 
    wavs = []
    for file in dir_files:
      if file.endswith(".wav"):
        wav = encoder.preprocess_audio(dir_path + "/" + file)
        wavs.append(wav)

    if len(wavs) > 0:
      print("[INFO] Multispeaker Synthesis - Generating embed given %d files for path %s..." % (len(wavs), dir_path))
      embed = encoder.embed_utterances(wavs)
      if embed_fpath is not None:
        np.save(embed_fpath, embed, allow_pickle = False)
      else:
        return embed
    else:
      print("[WARNING] Multispeaker Synthesis - No wav files found in %s" % dir_path)
    return None

  # Given a directory, for all subdirectory wav files, generate 
  # embeddings with the same name but with .npy suffixes. 
  # Ignores top level directory wavs. 
  def generate_subdirectory_embeds(self, parent_directory: Path):
    print("[INFO] Multispeaker Synthesis - Generating embeds for all subdirectories of %s" % parent_directory)
    for root, dirs, files in os.walk(parent_directory):
      for dir in dirs:
        full_dir = root + "/" + dir
        self.directory_embedding(full_dir, root + "/" + dir + ".npy")
        """
        full_dir = root + "/" + dir
        dir_files = os.listdir(full_dir)
        dir_files = [full_dir + "/" + file for file in dir_files] 

        for dir_file in dir_files:
          if dir_file.endswith(".wav"):
            print("[DEBUG] Multispeaker Synthesis - Embedding speaker in %s." % dir_file)
            self.single_embedding(wav_fpath = dir_file, embed_fpath = dir_file.replace(".wav", ""))
        """
    print("[INFO] Multispeaker Synthesis - All done!")


# Debug usage. 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("wav_fpath")
  args = parser.parse_args()

  speaker_encoder_fpath = Path("./production_models/speaker_encoder/model6/encoder.pt")

  wav_fpath = args.wav_fpath

  embedding = Embedding(speaker_encoder_fpath)
  embedding.single_embedding(wav_fpath, wav_fpath.replace(".wav",""))
  """
  # Ex) python production_embedding.py ../multispeaker_synthesis_speakers
  parser = argparse.ArgumentParser()
  parser.add_argument("parent_directory")
  args = parser.parse_args()

  speaker_encoder_fpath = Path("./production_models/speaker_encoder/model6/encoder.pt")

  embedding = Embedding(speaker_encoder_fpath)
  embedding.generate_subdirectory_embeds(args.parent_directory)
  """