#
# preprocess.py
#
# Speaker Encoder dataset preprocessing. Data used originates from 
# VoxCeleb1, VoxCeleb2, and Librispeech datasets and must be 
# preprocessed to ensure a median duration of around 4 seconds, 
# among other transformations. 
#
# All data should be combined into npy files in directories 
# organized by speaker. 

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import numpy as np

import speaker_encoder.audio_utils
from speaker_encoder.dataset_params import librispeech_datasets, anglophone_nationalites
from speaker_encoder.audio_params import * 

# The extensions we expect to find in our dataset
_AUDIO_EXTENSIONS = {"wav", "flac", "m4a", "mp3"}

# Dataset metadata file prefix, to be followed by dataset name + ".txt".
_dataset_log_prefix = "Log_"
_dataset_log_spacer = "==================================================================="
_preprocess_speaker_dirs_multiprocessing = 4

# Helper class for logging dataset metadata into a text file. This
# provides info like when the dataset was created, the parameter
# values, samples included, and dataset statistics. 
#
# To use, init the DatasetLog to create the file + add parameters. 
# Then, repeatedly call add_sample to fill the dict, before calling
# finalize to finish the file. 
class DatasetLog:
  # Given the root path + name of the dataset, create the file upon
  # initialization
  def __init__(self, root, name):
    # Replace any illegal / characters with _.
    self.text_file = open(Path(root, _dataset_log_prefix + "%s.txt" % name.replace("/", "_")), "w")
    self.sample_data = dict()

    # Write "header" information before logging stuff. 
    self.write_line(_dataset_log_spacer)
    self.write_line("Multispeaker Synthesis - Speaker Encoder Dataset")
    self.write_line(_dataset_log_spacer)
    start_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
    self.write_line("Dataset %s creation started: %s" % (name, start_time))
    self.write_line("")
    self._log_params()

  # For the sake of sanity, have a function that adds the \n for writing.
  def write_line(self, line):
    self.text_file.write("%s\n" % line)

  # Internal function for filing out the file at intit time. 
  def _log_params(self):
    self.write_line(_dataset_log_spacer)
    self.write_line("Parameter Values")
    self.write_line(_dataset_log_spacer)
    # For every single variable in the audio_params file, write it out. 
    import speaker_encoder.audio_params
    for param_name in (p for p in dir(speaker_encoder.audio_params) if not p.startswith("__")):
      value = getattr(speaker_encoder.audio_params, param_name)
      self.write_line("\t%s: %s" % (param_name, value))
    self.write_line("")

  # Submit a new sample to the dict. Takes in a variable amount of 
  # arguments + values. For each parameter, may store multiple 
  # values (stored in a list).
  def add_sample(self, **kwargs):
    for param_name, value in kwargs.items():
      if not param_name in self.sample_data:
        self.sample_data[param_name] = []
      self.sample_data[param_name].append(value)

  # Finishes the file. All samples should've been provided, first. 
  def finalize(self):
    self.write_line(_dataset_log_spacer)
    self.write_line("Dataset Statistics")
    self.write_line(_dataset_log_spacer)
    for param_name, values in self.sample_data.items():
      self.write_line("\t%s:" % param_name)
      self.write_line("\t\tmin %.3f, max %.3f" % (np.min(values), np.max(values)))
      self.write_line("\t\tmean %.3f, median %.3f" % (np.mean(values), np.median(values)))
    self.write_line("")

    # We're officially done. 
    end_time = str(datetime.now().strftime("%A %d %B %Y at %H:%M"))
    self.write_line("Dataset creation finished: %s" % end_time)
    self.text_file.close()

# Start the preprocessing for a new dataset, given the new dataset
# name + root directory of all datasets + output directory.
#
# Returns the full path to the dataset + created DatasetLog, or
# None, None.
def _init_preprocess_dataset(dataset_name, datasets_root, out_dir):
  dataset_root = datasets_root.joinpath(dataset_name)
  if not dataset_root.exists():
    print("[ERROR] Unable to find %s; skipping dataset %s." % (dataset_root, dataset_name))
    return None, None
  return dataset_root, DatasetLog(out_dir, dataset_name)

# We need to section off the data by speaker. Preprocess everything
# and write them all to npy arrays in by-speaker directories. Return
# the total duration of speaker audio. 
def _preprocess_speaker(speaker_dir: Path, datasets_root: Path, out_dir: Path, skip_existing: bool):
  # The speaker gets a unique name including dataset name (in the
  # nearly impossible chance that a celebrity voiced an audiobook.)
  speaker_name = "_".join(speaker_dir.relative_to(datasets_root).parts)

  # Create a new output directory for this speaker given the name.
  # Provide also, a txt file with a reference to each soure file. 
  speaker_out_dir = out_dir.joinpath(speaker_name)
  speaker_out_dir.mkdir(exist_ok=True)
  sources_fpath = speaker_out_dir.joinpath("_sources.txt")

  # Check to make sure it exists already (in case preprocessing 
  # started for this model but was stopped prematurely).
  if sources_fpath.exists():
    try:
      # It exists. Try to read it and get all files.
      with sources_fpath.open("r") as sources_file:
        existing_fnames = {line.split(",")[0] for line in sources_file}
    except:
      existing_fnames = {}
  else:
    existing_fnames = {}

  # Now let's recursively get all audio files for this speaker
  # in the dataset. 
  sources_file = sources_fpath.open("a" if skip_existing else "w")
  audio_durs = []
  # Grab all the files with all the accepted extensions.
  for extension in _AUDIO_EXTENSIONS:
    for in_fpath in speaker_dir.glob("**/*.%s" % extension):
      # Check if the target output file exists, which is the 
      # input filename but with .npy. If so, skip it (if allowed)
      out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts)
      out_fname = out_fname.replace(".%s" % extension, ".npy")
      if skip_existing and out_fname in existing_fnames:
        continue

      # Load and preprocess the waveform with our audio_utils. 
      # We expect a waveform (npy array)
      wav = speaker_encoder.audio_utils.preprocess_audio(in_fpath)
      if len(wav) == 0:
        print("[WARNING] Preprocess - Preprocessed audio empty for file: '%s'," % in_fpath)
        continue

      # Create the mel spectogram. If it's too short, remove it. 
      frames = speaker_encoder.audio_utils.wav_to_mel_spectogram(wav)
      if len(frames) < partials_n_frames:
        print("[WARNING] Preprocess - Mel spectogram too short. Extracted from file: '%s'," % in_fpath)
        continue

      # All done; write the spectogram accordingly. 
      out_fpath = speaker_out_dir.joinpath(out_fname)
      np.save(out_fpath, frames)
      sources_file.write("%s,%s\n" % (out_fname, in_fpath))
      audio_durs.append(len(wav) / sampling_rate)
  
  sources_file.close()
  return audio_durs

# Once all audio has been written, time to segment the spectograms
# into utterances. 
def _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger):
  print("[INFO] Preprocess - Preprocessing dataset %s for %d speakers." % (dataset_name, len(speaker_dirs)))

  # For each speaker, process the utterances. Use a "working function"
  # variable for ease of use having specified 3/4 params.
  work_fn = partial(_preprocess_speaker, datasets_root=datasets_root, 
                    out_dir=out_dir, skip_existing=skip_existing)
  # Yay for multiprocessing - batch size of 4. 
  with Pool(_preprocess_speaker_dirs_multiprocessing) as pool:
    tasks = pool.imap(work_fn, speaker_dirs)
    for sample_durs in tqdm(tasks, dataset_name, len(speaker_dirs), unit="speakers"):
      for sample_dur in sample_durs:
        logger.add_sample(duration=sample_dur)
  
  # All done. 
  logger.finalize()
  print("[INFO] Preprocess - Dataset %s preprocessing complete.\n" % dataset_name)

# For LibriSpeech, we'll handle the various subdirectories as specified
# in dataset_params.py. Straightforward.
def preprocess_librispeech(datasets_root: Path, out_dir: Path, skip_existing = False):
  for dataset_name in librispeech_datasets["train"]["other"]:
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    if not dataset_root:
      return # We've already logged the error.
    
    # Go through all the speakers. 
    speaker_dirs = list(dataset_root.glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)

# VoxCeleb1 is the most complicated of the three datasets - we want to 
# take advantage of the nationality info of the speaker to reduce the #
# of foreign languages in our datset (can't do anything for vox2).
def preprocess_voxceleb1(datasets_root: Path, out_dir: Path, skip_existing = False):
  dataset_name = "VoxCeleb1" # This is expected.
  dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
  if not dataset_root:
    return # We've already logged the error. 

  # Filter out the speakers by nationality via meta file. We expect
  # this to be here and will die if it isn't. 
  metadata = None
  try:
    with dataset_root.joinpath("vox1_meta.csv").open("r") as metafile:
      # The metadata file has only one column with everything squished
      # inside. 
      metadata = [line.split("\t") for line in metafile][1:]
  except:
    print("[ERROR] Preprocess - Unable to find vox1_meta.csv. Has it been placed in the VoxCeleb1 folder?")
    return
  
  # Work through the metadata to get the speakers we want. 
  nationalities = {line[0]: line[3] for line in metadata}
  keep_speaker_ids = [speaker_id for speaker_id, nationality in nationalities.items() if
                      nationality.lower() in anglophone_nationalites]
  print("[INFO] Preprocess - VoxCeleb1 using samples from %d presumed anglophone speakers out of %d speakers." %
        (len(keep_speaker_ids), len(nationalities)))
  
  # Now let's grab the speaker directories for these speakers.
  speaker_dirs = dataset_root.joinpath("wav").glob("*")
  speaker_dirs = [speaker_dir for speaker_dir in speaker_dirs if
                  speaker_dir.name in keep_speaker_ids]
  print("[INFO] Preprocess - VoxCeleb1 found %d anglophone speakers on disk, with %d missing (Some missing is expected)." %
        (len(speaker_dirs), len(keep_speaker_ids) - len(speaker_dirs)))
  
  # And finally, let's kick the preprocessing off for this dataset.
  _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)

# VoxCeleb2 is as easy as LibriSpeech - kick off processing for all
# speakers. 
def preprocess_voxceleb2(datasets_root: Path, out_dir: Path, skip_existing = False):
  dataset_name = "VoxCeleb2" # This is expected.
  dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
  if not dataset_root:
    return # We've already logged the error.
  
  # Go through all the speakers.
  speaker_dirs = list(dataset_root.joinpath("dev", "aac").glob("*"))
  _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, skip_existing, logger)