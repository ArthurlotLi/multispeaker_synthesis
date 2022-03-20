#
# preprocess.py
#
# Synthesizer dataset preprocessing. Data used originates from 
# LibriSpeech-Clean dataset.

from multiprocessing import Pool
from re import M
import synthesizer.audio_utils as audio
import speaker_encoder.inference as encoder
import utils.logmmse as logmmse # Third party code for profiling noise in samples.

from functools import partial
from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa

# Principal function. Preprocesses the LibriSpeech dataset to suit
# our needs. 
def preprocess_dataset(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool,
                       hparams, no_alignments: bool, datasets_name: str, subfolders: str):
  # Gather the input directories and verify they all exist.
  dataset_root = datasets_root.joinpath(datasets_name)
  input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
  print("[INFO] Preprocess - Using data from: ")
  print(" ".join(map(str, input_dirs)))
  assert all(input_dir.exists() for input_dir in input_dirs)

  # Create output directories for each output file type. 
  out_dir.joinpath("mels").mkdir(exist_ok=True)
  out_dir.joinpath("audio").mkdir(exist_ok=True)

  # Create a metadata file that we'll use when loading the data. 
  metadata_fpath = out_dir.joinpath("train.txt")
  metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding = "utf-8")

  # Now preprocess the dataset. Execute the preprocess_speaker function
  # with multiprocessing, with the n_processes having been specified
  # by the user.
  speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
  func = partial(preprocess_speaker, out_dir=out_dir, skip_existing=skip_existing,
                 hparams=hparams, no_alignments=no_alignments)
  job = Pool(n_processes).imap(func, speaker_dirs)
  for speaker_metadata in tqdm(job, datasets_name, len(speaker_dirs), unit="speakers"):
    for metadatum in speaker_metadata:
      metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
  metadata_file.close()

  # Preprocessing done. Let's verify the metadata file and print out some
  # statistics. 
  with metadata_fpath.open("r", encoding= "utf-8") as metadata_file:
    metadata = [line.split("|") for line in metadata_file]
  mel_frames = sum([int(m[4]) for m in metadata])
  timesteps = sum([int(m[3]) for m in metadata])
  sample_rate = hparams.sample_rate
  hours = (timesteps / sample_rate) / 3600

  print("[INFO] Preprocess - Preprocessed dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
        (len(metadata), mel_frames, timesteps, hours))
  print("[INFO] Preprocess - Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
  print("[INFO] Preprocess - Max mel frames length: %d" % max(int(m[4]) for m in metadata))
  print("[INFO] Preprocess - Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))

# For each speaker directory, get the audio, get the corresponding 
# transcript, and generate samples (mel spectogram with text).
def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, 
                       hparams, no_alignments: bool):
  metadata = []
  for book_dir in speaker_dir.glob("*"):
    if no_alignments:
      # Not using alignment. Gather utterance audios and text.
      # Note that LibriTTS (which we may try) uses .wav, but just in case
      # we allow for othe rextensions. 
      extensions = ["*.wav", "*.flac", "*.mp3"]
      for extension in extensions:
        wav_fpaths = book_dir.glob(extension)

        for wav_fpath in wav_fpaths:
          # Load the audio waveform at the specified sample rate.
          wav, _ = librosa.load(str(wav_fpath), sr=hparams.sample_rate)
          if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
          
          # Gather the corresponding text with .txt suffix. 
          text_fpath = wav_fpath.with_suffix(".txt")
          if not text_fpath.exists():
            # Check for .normalized.txt (LibriTTS specific)
            text_fpath = wav_fpath.with_suffix(".normalized.txt")
            assert text_fpath.exists()
          
          # Processs the file. Remove all quotes + strip whitespace.
          with text_fpath.open("r") as text_file:
            text = "".join([line for line in text_file])
            text = text.replace("\"", "")
            text = text.strip()
          
          # Process the utterance. 
          metadata.append(process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name), 
                                           skip_existing, hparams))
    else:
      # Using alignment. Process the alignment file (We generate one
      # for LibriSpeech). Gather utterance audios and text.
      try:
        alignments_fpath = next(book_dir.glob("*.alignment.txt"))
        with alignments_fpath.open("r") as alignments_file:
          alignments = [line.rstrip().split(" ") for line in alignments_file]
      except StopIteration:
        # A few alignment files will be missing; just skip them. 
        print("[WARNING] Preprocess - Unable to find alignment file for %s. Skipping." % speaker_dir)
        continue
      
      # Iterate through every entry in the alignments file. 
      for wav_fname, words, end_times in alignments:
        wav_fpath = book_dir.joinpath(wav_fname + ".flac")
        assert wav_fpath.exists()
        # Preprocess the alignment by removing quotations, split on 
        # commas. 
        words = words.replace("\"", "").split(",")
        end_times = list(map(float, end_times.replace("\"", "").split(",")))

        # Process each sub-utterance; split on silences + process
        # each split utterance. 
        wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams)
        for i, (wav, text) in enumerate(zip(wavs, texts)):
          sub_basename = "%s_%02d" % (wav_fname, i)
          metadata.append(process_utterance(wav, text, out_dir, sub_basename, 
                                            skip_existing, hparams))

  # All done. Return metadata information. 
  return [m for m in metadata if m is not None]

# Splitting on silences that are longer than a specified duration. 
# Expects aligned utterances. 
def split_on_silences(wav_fpath, words, end_times, hparams):
  # Load the audio waveform at the specified sample rate
  wav, _ = librosa.load(str(wav_fpath), sr=hparams.sample_rate)
  if hparams.rescale:
    wav = wav / np.abs(wav).max() * hparams.rescaling_max

  # Convert words, start times and end times into numpy arrays.
  # Make sure they are all of the same length, and the words
  # contains a leading + trailing empty char.
  words = np.array(words)
  start_times = np.array([0.0] + end_times[:-1])
  end_times = np.array(end_times)
  assert len(words) == len(end_times) == len(start_times)
  assert words[0] == "" and words[-1] == ""

  # Find pauses that are too long with a bit mask. 
  mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
  mask[0] = mask[-1] = True
  breaks = np.where(mask)[0]

  # Profile the noise from silences and perform noise reduction on
  # the waveform using the logmmse implementation. This uses the
  # silence in the beginning of the audio. If the length of the 
  # original snippet is long enough, profile noise and denoise. 
  silence_times = [[start_times[i], end_times[i]] for i in breaks]
  silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
  noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
  if len(noisy_wav) > hparams.sample_rate * 0.02:
    profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
    wav = logmmse.denoise(wav, profile, eta=0)
  
  # Re-attach segments that turned out to be too short.
  segments = list(zip(breaks[:-1], breaks[1:]))
  segment_durations = [start_times[end] - end_times[start] for start, end in segments]
  i = 0
  while i < len(segments) and len(segments) > 1:
    if segment_durations[i] < hparams.utterance_min_duration:
      # Attempt to reattach the segment to left or right.
      left_duration = float("inf") if i == 0 else segment_durations[i - 1]
      right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
      joined_duration = segment_durations[i] + min(left_duration, right_duration)

      # Don't reattach if it causes the joined utterance to exceed
      # our maximum segment length. Dump this segment. 
      if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
        i += 1
        continue
    
      # Reattach the segment with the neighbor of the shortest duration.
      j = i - 1 if left_duration <= right_duration else i
      segments[j] = (segments[j][0], segments[j + 1][1])
      segment_durations[j] = joined_duration
      del segments[j + 1], segment_durations[j + 1]
    else:
      i += 1
  
  # Split the utterance. 
  segment_times = [[end_times[start], start_times[end]] for start, end in segments]
  segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
  wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
  texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

  return wavs, texts

# Processing a waveform + text. 
# 
# A warning from the github author: 
# For you not to lose your head if you ever wish to change things here or implement your own
# synthesizer.
# - Both the audios and the mel spectrograms are saved as numpy arrays
# - There is no processing done to the audios that will be saved to disk beyond volume
#   normalization (in split_on_silences)
# - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
#   is why we re-apply it on the audio on the side of the vocoder.
# - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
#   without extra padding. This means that you won't have an exact relation between the length
#   of the wav and of the mel spectrogram. See the vocoder data loader.
def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename:str,
                      skip_existing: bool, hparams):
  # Skip existing utterances if needed
  mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
  wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
  if skip_existing and mel_fpath.exists() and wav_fpath.exists():
    return None
  
  # Trim silence using the speaker encoder function.
  if hparams.trim_silence:
    wav = encoder.preprocess_audio(wav, normalize=False, trim_silence=True)

  # Skip utterances that came out as too short. 
  if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
    return None
  
  # Compute the mel spectogram
  mel_spectogram = audio.melspectogram(wav, hparams).astype(np.float32)
  mel_frames = mel_spectogram.shape[1]

  # Skip utterances that came out as too long.
  if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
    return None
  
  # Write the spectogram, embed, and audio to disk. 
  np.save(mel_fpath, mel_spectogram.T, allow_pickle=False)
  np.save(wav_fpath, wav, allow_pickle=False)

  # Return a tuple with the metadata info. 
  return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, len(wav), mel_frames, text

# Derive embeddings by running them through our painstakingly trained
# speaker encoder model. 
def embed_utterance(fpaths, encoder_model_fpath):
  if not encoder.is_loaded():
    encoder.load_model(encoder_model_fpath)

  # Compute the speaker embedding. 
  wav_fpath, embed_fpath = fpaths
  wav = np.load(wav_fpath)
  wav = encoder.preprocess_audio(wav)
  embed = encoder.embed_utterance(wav)
  np.save(embed_fpath, embed, allow_pickle = False)

# Derive all embeddings.
def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
  wav_dir = synthesizer_root.joinpath("audio")
  metadata_fpath = synthesizer_root.joinpath("train.txt")
  assert wav_dir.exists() and metadata_fpath.exists()
  embed_dir = synthesizer_root.joinpath("embeds")
  embed_dir.mkdir(exist_ok = True)

  # Gather the input wav filepath and target output embed filepath.
  with metadata_fpath.open("r") as metadata_file:
    metadata = [line.split("|") for line in metadata_file]
    fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]
  
  # Multiprocessing. Embed the utterance in separate threads.
  # NOTE: Multiprocessing here isn't great - disk I/O is the bottleneck. 
  func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
  job = Pool(n_processes).imap(func, fpaths)
  list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))