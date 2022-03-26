#
# production_speaker_verification.py
#
# Utilizes the speaker_encoder model on the orginal speaker_verification
# task. Allows for tasks such as finding all files in a folder that 
# deviate significantly from the embeddings of the other utterances. 

from production_embedding import Embedding

from tqdm import tqdm
from numpy import linalg as LA
from pathlib import Path
import argparse
import os

_audio_suffix = ".wav"

def verify_singular_directory(wavs_fpath, speaker_encoder_fpath, l2_tolerance):
  """
  Given a path to a directory, calculate the embeddings of all files.
  Given those, compare the embeddings and flag wav files that 
  significantly deviate from the others. 

  Returns tuples of filenames that seem off to the model. 
  """
  if not os.path.exists(wavs_fpath):
    print("[ERROR] Speaker Verification - Unable to find %s." % wavs_fpath)
    return
  
  data_contents = os.listdir(wavs_fpath)
  wav_files = []
  for file in data_contents:
    if file.endswith(_audio_suffix):
      wav_files.append(file)

  data_count = len(wav_files)
  print("[INFO] Speaker Verification - found %d %s files in folder %s." % (data_count, _audio_suffix, wavs_fpath))

  embedding = Embedding(speaker_encoder_fpath)

  print("[INFO] Speaker Verification - Embedding all files.")
  embeddings = []
  # TODO: It would be way faster to do batch embedding. 
  for i in tqdm(range(0, data_count), "Embedding Audio"):   
    filename = wav_files[i]
    file_path = wavs_fpath + "/" + filename
    embeds = embedding.single_embedding(file_path)
    embeddings.append((filename, embeds))
  
  # Given all the embeddings, calculate the difference between all of them. 
  # TODO: There is a lot of duplicated work here. 
  print("[INFO] Speaker Verification - Comparing %d embeddings..." % len(embeddings))
  embedding_norms = []
  for filename, embeds in embeddings:
    combined_norms = 0
    embeddings_compared = 0
    for test_filename, test_embeds in embeddings:
      if test_filename != filename:
        # Compare. Get the diffs vector.
        diffs = embeds - test_embeds
        # Get the L2 norm of this vector.
        l2_norm = LA.norm(diffs)
        combined_norms += l2_norm
        embeddings_compared += 1

    # Average embeddings.
    if embeddings_compared != 0:
      combined_norms = combined_norms / embeddings_compared
      embedding_norms.append((combined_norms, filename, embeds))
    else:
      embedding_norms.append((0, filename, embeds))

  reported_files = []
  report_threshold = l2_tolerance
  print("[INFO] Speaker Verification - Done. Samples > %.2f:" % report_threshold)
  #for combined_norms, filename, embeds in embedding_norms:
    #print("       %.2f - %s" % (combined_norms, filename))
  for combined_norms, filename, embeds in embedding_norms:
    if combined_norms > report_threshold:
      print("       %.2f - %s" % (combined_norms, filename))
      reported_files.append((combined_norms, filename))
  
  return reported_files

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("wavs_fpath")
  args = parser.parse_args()

  l2_tolerance = 1.14
  model_num = "model1"
  speaker_encoder_fpath = Path("./production_models/speaker_encoder/"+model_num+"/encoder.pt")

  verify_singular_directory(args.wavs_fpath, speaker_encoder_fpath, l2_tolerance)