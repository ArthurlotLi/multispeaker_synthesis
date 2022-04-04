#
# production_verified_embedding.py
#
# Given speakers with singular samples in subfolders as well as
# paths to full speaker data, gather samples that are similar to the
# initial sample by comparing L2 normalized distances. 
#
# Essentially using speaker verification to execute speaker
# embedding to avoid extremely dissimilar samples that may potentially
# lead to "embedding mush" in the act of averaging embeddings.
#
# The default configuration takes every speaker inside of your main dir
# and generates a .npy embedding with the single largest group of wavs
# that are close to each other within the specified L2 norm tolerance.
# A .wav for the selected target audio file is also generated for 
# reference (So you can see what audio file is the embedding trying to
# generally represent). 
#
# Make sure that your directory of ALL audio data is structured like so:
#  - Main dir
#    - SPEAKER
#       - <wavs>
#    - SPEAKER
#    ...
#    <> This is where the product .npy files and .wav files will be produced.
#
# Usage: 
# python production_verified_embedding.py ../multispeaker_synthesis_speakers 0.35

from functools import partial
from pathlib import Path
import argparse
import os

from production_embedding import Embedding


from multiprocessing import Pool
from tqdm import tqdm
from numpy import linalg as LA
from pathlib import Path
import argparse
import os
import shutil

_audio_suffix = ".wav"
_output_wavs = False
_temp_folder_name = "temp"
_n_processes = 10
_use_multiprocessing = True
_minimum_samples = 5 # Under this amount, we'll up the L2 tolerance. 
_minimum_samples_l2_additional = 0.5 # Accept A LOT MORE samples.

def embed_speakers(full_data_dir:Path, encoder_fpath: Path, 
                   l2_tolerance: float, target_wavs: bool):
  """
  Given a directory of speakers, go through each directory and execute
  verified embedding. There may be an arbitrary number of wavs in the
  folder to compute a verified embedding for. 

  Each speaker directory should be formatted as such:
  - SPEAKER
    - full_data
    - <wav_name.wav>

  The result will be the following for each speaker:
  - SPEAKER
    - full_data
    - <wav_name> (dir of used wavs)
    - <wav_name.wav>
    - <wav_name.npy>
  """
  if not full_data_dir.exists:
    print("[ERROR] Verified Embedding - Unable to find %s." % full_data_dir)
    return

  # Compute the speaker embedding for each.
  args = []
  for item in os.listdir(str(full_data_dir)):
    item_path = str(full_data_dir.joinpath(item))
    if os.path.isdir(item_path):

      # Get the directory for full data for this speaker. We expect this
      # to be in the full_data_dir directly under the same name. 
      speaker_full_data = full_data_dir.joinpath(item)
      assert speaker_full_data.exists()
      
      if _use_multiprocessing is False:
        print("[INFO] Verified Embedding - Computing embeddings for speaker \"%s\"" % item)
        if target_wavs:
          speaker_verified_embedding_for_targets(Path(item_path), speaker_full_data, l2_tolerance, encoder_fpath)
        else:
          args = (Path(item_path), speaker_full_data, l2_tolerance, encoder_fpath)
          speaker_verified_embedding_single(args)
      else:
        args.append((Path(item_path), speaker_full_data, l2_tolerance, encoder_fpath))

  if _use_multiprocessing:
    func = partial(speaker_verified_embedding_single, verbose=False)
    job = Pool(_n_processes).imap(func, args)
    list(tqdm(job, "Embedding", len(args), unit="speakers"))

  print("[INFO] Verified Embedding - All done. Goodnight...")

# NOTE: Obsolete, but kept around just in case. 
def speaker_verified_embedding_for_targets(speaker_dir: Path, full_data: Path, l2_tolerance:float, encoder_fpath: Path):
  """
  First, compute embeddings for all data in the full_data folder and
  store them linked to their wav filepaths. 
  
  Then, execute verification to get the list of wavs to exclude for 
  each wav. Take all accepted wavs and copy into a new directory under 
  the name of the wav. Repeat this for every source wav. (Delete the
  existing files in the directory if it exists.)

  For each generated directory, generate a production embedding for
  final products and place in the speaker dir next to the original 
  directory. 
  """
  # Load the embedding. 
  embedding = Embedding(encoder_fpath)

  if not speaker_dir.exists:
    print("[ERROR] Verified Embedding - Unable to find %s." % speaker_dir)
    assert(False)
  if not full_data.exists:
    print("[ERROR] Verified Embedding - Unable to find %s." % full_data)
    assert(False)
  
  # Compute embeddings for all files in full_data folder and store them
  # in a dict. 
  dir_files = os.listdir(str(full_data))

  # Compute the speaker embeddings and store in a dict keyed by filename.
  print("[INFO] Verified Embedding - Computing %d embeddings at %s." % (len(dir_files),full_data))
  embeddings = []
  for file in tqdm(dir_files):
    if file.endswith(_audio_suffix):
      wav_full_path = full_data.joinpath(file)
      embeds = embedding.single_embedding(str(wav_full_path))
      embeddings.append((wav_full_path, embeds))
  
  # Get the list of wav files to generate embeddings for. 
  target_wavs = {}
  dir_files = os.listdir(str(speaker_dir))
  for file in dir_files:
    if file.endswith(_audio_suffix):
      wav_full_path = speaker_dir.joinpath(file)
      target_wavs[file] = embedding.single_embedding(str(wav_full_path))

  # Execute speaker verification. 
  target_wav_accepted = {}
  target_wav_rejected = {}
  for target_wav_name in target_wavs:
    print("[INFO] Verified Embedding - Comparing %d embeddings against %s." % (len(embeddings), target_wav_name))
    embeds = target_wavs[target_wav_name]
    accepted = []
    rejected = []

    for filename, test_embeds in embeddings:
      # Compare. Get the diffs vector.
      diffs = embeds - test_embeds
      # Get the L2 norm of this vector.
      l2_norm = LA.norm(diffs)

      # If within the threshold, accept.
      if l2_norm <= l2_tolerance:
        accepted.append(filename)
      else:
        rejected.append(filename)
    
    target_wav_accepted[target_wav_name] = accepted
    target_wav_rejected[target_wav_name] = rejected
  
  # We now have lists of wavs to use for each target wav. For each target
  # wav, generate a new folder and copy the accepted wavs into that folder. 
  for target_wav_name in target_wavs:
    target_wav_wavs = speaker_dir.joinpath(target_wav_name.replace(_audio_suffix, ""))
    if target_wav_wavs.exists():
      # Remove everything in that directory. 
      print("[INFO] Verified Embedding - Removing existing wavs in %s." % target_wav_wavs)
      for file in os.listdir(str(target_wav_wavs)):
        os.remove(str(target_wav_wavs.joinpath(file)))
    else:
      os.mkdir(str(target_wav_wavs))
    
    # Directory is ready. Copy all files. 
    print("[INFO] Verified Embedding - Copying %d accepted files to %s." % (len(target_wav_accepted[target_wav_name]),target_wav_wavs))
    for file in tqdm(target_wav_accepted[target_wav_name]):
      shutil.copyfile(str(file), str(target_wav_wavs.joinpath(file.name)))

    # Now let's execute the embedding.
    print("[INFO] Verified Embedding - Creating product embedding for %s." % target_wav_name)
    embedding.directory_embedding(str(target_wav_wavs), str(speaker_dir.joinpath(target_wav_name.replace(_audio_suffix, ""))))

    if _output_wavs is False:
      # Remove everything in that directory. 
      print("[INFO] Verified Embedding - Removing existing wavs in %s." % target_wav_wavs)
      for file in os.listdir(str(target_wav_wavs)):
        os.remove(str(target_wav_wavs.joinpath(file)))

    # As a final step, we can copy the files from the rejected side. 
    # I'm a hurry, so this is copied code. 
    if _output_wavs:
      target_wav_wavs = speaker_dir.joinpath(target_wav_name.replace(_audio_suffix, "_removed"))
      if target_wav_wavs.exists():
        # Remove everything in that directory. 
        print("[INFO] Verified Embedding - Removing existing wavs in %s." % target_wav_wavs)
        for file in os.listdir(str(target_wav_wavs)):
          os.remove(str(target_wav_wavs.joinpath(file)))
      else:
        os.mkdir(str(target_wav_wavs))
      
      # Directory is ready. Copy all files. 
      print("[INFO] Verified Embedding - Copying %d rejected files to %s." % (len(target_wav_rejected[target_wav_name]),target_wav_wavs))
      for file in tqdm(target_wav_rejected[target_wav_name]):
        shutil.copyfile(str(file), str(target_wav_wavs.joinpath(file.name)))

def speaker_verified_embedding_single(args, verbose = True):
  """
  Similar to the above, however does not require manual selection of
  a wav file. It grabs the largest group of utterances according to 
  the l2 tolerance and uses it to generate a product embedding for
  the speaker. 
  """
  speaker_dir = args[0] 
  full_data = args[1] 
  l2_tolerance = args[2] 
  encoder_fpath = args[3]

  # Load the embedding. 
  embedding = Embedding(encoder_fpath)

  if not speaker_dir.exists:
    print("[ERROR] Verified Embedding - Unable to find %s." % speaker_dir)
    assert(False)
  if not full_data.exists:
    print("[ERROR] Verified Embedding - Unable to find %s." % full_data)
    assert(False)
  
  # Compute embeddings for all files in full_data folder and store them
  # in a dict. 
  dir_files = os.listdir(str(full_data))

  # Compute the speaker embeddings and store in a dict keyed by filename.
  if verbose: print("[INFO] Verified Embedding - Computing %d embeddings at %s." % (len(dir_files),full_data))
  embeddings = []
  if verbose: enum = tqdm(dir_files)
  else: enum = dir_files
  for file in enum:
    if file.endswith(_audio_suffix):
      wav_full_path = full_data.joinpath(file)
      embeds = embedding.single_embedding(str(wav_full_path))
      embeddings.append((wav_full_path, embeds))

  def get_best_wav(embeddings, l2_tolerance):
    best_wav_filename = None
    best_wav_num = -1
    target_wav_accepted = None
    if verbose: print("[INFO] Verified Embedding - Comparing embeddings.")
    if verbose: enum = tqdm(embeddings)
    else: enum = embeddings
    for target_filename, target_embeds in enum:
      accepted = [target_filename]
      rejected = []

      for filename, test_embeds in embeddings:
        if filename != target_filename:
          # Compare. Get the diffs vector.
          diffs = target_embeds - test_embeds
          # Get the L2 norm of this vector.
          l2_norm = LA.norm(diffs)

          # If within the threshold, accept.
          if l2_tolerance == -1 or l2_norm <= l2_tolerance:
            accepted.append(filename)
          else:
            rejected.append(filename)

      # If this wav has the most amount of candidates, save it. 
      if best_wav_num < len(accepted):
        best_wav_num = len(accepted)
        best_wav_filename = target_filename
        target_wav_accepted = {}
        target_wav_accepted[target_filename] = accepted
  
    return best_wav_filename, target_wav_accepted, best_wav_num

  best_wav_filename, target_wav_accepted, best_wav_num = get_best_wav(embeddings, l2_tolerance)
  if best_wav_num < _minimum_samples and len(embeddings) > _minimum_samples:
    old_target_wav = best_wav_num
    # Try again, one more time, with a far greater tolerance.
    best_wav_filename, target_wav_accepted, best_wav_num = get_best_wav(embeddings, l2_tolerance + _minimum_samples_l2_additional)
    print("[WARNING] Verified Embedding - %d accepted wavs for %s is too little! Increased tolerance and got %d samples out of %d embeds." % (old_target_wav, speaker_dir.name, best_wav_num, len(embeddings)) )

  assert best_wav_filename is not None
  
  # We now have lists of wavs to use for each target wav. For each target
  # wav, generate a new folder and copy the accepted wavs into that folder. 
  target_wav_name = _temp_folder_name + _audio_suffix

  target_wav_wavs = speaker_dir.joinpath(_temp_folder_name)
  if target_wav_wavs.exists():
    # Remove everything in that directory. 
    if verbose: print("[INFO] Verified Embedding - Removing existing wavs in %s." % target_wav_wavs)
    for file in os.listdir(str(target_wav_wavs)):
      os.remove(str(target_wav_wavs.joinpath(file)))
  else:
    os.mkdir(str(target_wav_wavs))

  # Copy the accepted central file for reference to the output directory. 
  shutil.copyfile(str(best_wav_filename), str(speaker_dir.parents[0].joinpath(speaker_dir.name)) + _audio_suffix)
  
  # Directory is ready. Copy all files. 
  if verbose: print("[INFO] Verified Embedding - Copying %d accepted files to %s." % (len(target_wav_accepted[best_wav_filename]), target_wav_wavs))
  if verbose: enum = tqdm(target_wav_accepted[best_wav_filename])
  else: enum = target_wav_accepted[best_wav_filename]
  for file in enum:
    shutil.copyfile(str(file), str(target_wav_wavs.joinpath(file.name)))

  # Now let's execute the embedding.
  if verbose: print("[INFO] Verified Embedding - Creating product embedding for %s." % target_wav_name)
  embedding.directory_embedding(str(target_wav_wavs), str(speaker_dir.parents[0].joinpath(speaker_dir.name)))

  if _output_wavs is False:
    # Remove everything in that directory. 
    if verbose: print("[INFO] Verified Embedding - Removing existing wavs in %s." % target_wav_wavs)
    for file in os.listdir(str(target_wav_wavs)):
      os.remove(str(target_wav_wavs.joinpath(file)))
    shutil.rmtree(str(target_wav_wavs))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("full_data_dir")
  parser.add_argument("l2_tolerance")
  args = parser.parse_args()

  # The following comments are relative to a neutral recording.
  l2_tolerance = float(args.l2_tolerance)

  #l2_tolerance = 1.14 # For extreme differences (yelling, etc)
  #l2_tolerance = 1.00 # For considerable differences (incredulous, sad, soft)
  #l2_tolerance = 0.80 # For noticable differences (different tones of neutral)
  model_num = "model6"
  speaker_encoder_fpath = Path("./production_models/speaker_encoder/"+model_num+"/encoder.pt")

  full_data_dir = Path(args.full_data_dir)

  # Use False if you don't intend to hand-pick a wav and place it in the folder. 
  target_wavs = False

  embed_speakers(full_data_dir, speaker_encoder_fpath, l2_tolerance, target_wavs)