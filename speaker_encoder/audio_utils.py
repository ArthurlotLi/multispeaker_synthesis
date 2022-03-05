#
# audio_utils.py
#
# Provides various methods relevant to audio file management and
# content preprocessing. Used for preprocessing and inference. 
# Note files provided may use a variety of extensions, not just
# .wav.
#
# The main methods you'd be using from here are 
# wav_to_mel_spectogram and preprocess_audio.

from speaker_encoder.audio_params import *

from scipy.ndimage.morphology import binary_dilation
from pathlib import Path
from typing import Optional, Union # Fun fact, a really cool way of forcing types.
import webrtcvad
import numpy as np
import librosa
import struct

# Used in trim_long_silences - largest number in int16.
int16_max = (2 ** 15) - 1

# Simple derivation of a mel spectogram from a wav for use for the
# speaker encoder given a waveform (that should be preprocessed!).
# NOT a log spectogram. Uses audio_params for constants.
#
# Returns an nparray of the derived mel spectogram. 
def wav_to_mel_spectogram(wav):
  frames = librosa.feature.melspectrogram(
    wav,
    sampling_rate,
    n_fft = int(sampling_rate*mel_window_length / 1000), 
    hop_length = int(sampling_rate * mel_window_step / 1000),
    n_mels = mel_n_channels
  )
  return frames.astype(np.float32).T

# The one-and-done function that handles preprocessing an audio
# file. For the speaker encoder, we have three preprocessing steps
# to take for each audio sample:
#   1. Match sampling rate if necessary (16kHz)
#   2. Trim out silences
#   3. Normalize audio to account for loud/soft samples
#
# Returns the preprocessed waveform. 
# 
# Fun fact, I've never used the typing package. It's pretty dope. 
# Expects either the audio file path or waveform (str/path) or nparray.
# Optionally may provide sampling rate (original, before preprocessing) 
# and/or stop normalization/trim.
def preprocess_audio(audio_location_or_wav: Union[str, Path, np.ndarray], 
                     source_sr: Optional[int] = None,
                     normalize: Optional[bool] = True,
                     trim_silence: Optional[bool] = True):

  # Load the file if given a path. If not given a path, we were given
  # a npy array of the waveform itself.
  wav = None
  if isinstance(audio_location_or_wav, str) or isinstance(audio_location_or_wav, Path):
    wav, source_sr = librosa.load(str(audio_location_or_wav), sr=None)
  else:
    wav = audio_location_or_wav

  # Resampling to match the expected sampling rate in audio_params if
  # necessary.
  if source_sr is not None and source_sr != sampling_rate:
    wav = librosa.resample(wav, source_sr, sampling_rate)

  # Apply Preprocessing. Audio normalization + Voice Activity Detection
  # (VAD, aka silence trimming)
  if normalize is True:
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
  
  if trim_silence:
    wav = trim_long_silences(wav)
  
  # All done.
  return wav

# Given a waveform and a targeted volume, Normalize it. Users may
# elect to restrict operations to increasing/decreasing volume only.
def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
  if increase_only and decrease_only:
    # Side note, I gotta do this more often. Way cleaner than prints...
    raise ValueError("Both increase/decrease only options may not be specified together!")

  # Get the difference necessary: log the mean of the doubled waveform.
  dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))

  # Don't make any changes if it violates any specified restrictions
  if(dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
    return wav

  # Apply the transformation. 
  return wav * (10 ** (dBFS_change / 20))
    
# Utilizes Voice Activation Detection to detect and trim periods of
# silence in the provided waveform. 
# 
# To do this, we first use webrtcvad to get binary flags over the 
# audio corresponding to whether the segment is voiced. Next, we 
# get a moving average of these flags, smoothing out spikes, before 
# binarizing again. The flags are then dialated with a kernel size 
# of s + 1, with s default to 0.2. 
#
# This entire process results in a nice and tidy segment that is
# entirely valuable (in theory) for the model to train on. 
#
# Takes in the wav file, and return sthe same waveform with any and
# all silences trimmed (with a length <= original width, naturally).
def trim_long_silences(wav):
  # Voice Activation Detection window size.
  samples_per_window = (vad_window_length * sampling_rate) // 1000

  # We need to cut off a little bit to match the window size (or a
  # multiple of it). Cut off what we can't process. 
  wav = wav[:len(wav) - (len(wav) % samples_per_window)]

  # Convert the waveform into a tractable 16-bit mono PCM using
  # struct.
  pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

  # Now we grab the binary flags by executing voice activation
  # detection using webrtcvad. Iterate through the wav and, for
  # each window, get a flag saying whether it is active or not.
  voice_flags = []
  vad = webrtcvad.Vad(mode=3)
  for window_start in range(0, len(wav), samples_per_window):
    window_end = window_start + samples_per_window
    voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                     sample_rate=sampling_rate))
  
  # Given our flags, apply a moving average to remove spikes
  # in the voice_flags array. We'll use a helper function for
  # this - given the array and the width of the moving average,
  # return the array with the moving average applied. 
  def moving_average(array, width):
    # Pad the array on both sides with (w-1//2) length zero vectors.
    # This is to help the moving average on the ends. 
    array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
    # Cumulative sum of elements along a given axis. 
    ret = np.cumsum(array_padded, dtype=float)
    # Clean up the length of the array. 
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width

  # Apply the moving average function. Use audio_params for the width.
  audio_mask = moving_average(voice_flags, vad_moving_average_width)
  # Renormalize back into boolean flags from our moving average.
  audio_mask = np.round(audio_mask).astype(np.bool)

  # Final step! Dilate the voiced regions to a specified tolerance 
  # of consecutive silence segments. 
  audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
  audio_mask = np.repeat(audio_mask, samples_per_window)

  # We're done here. Return the trimmed waveform, with a flag
  # for sanity check purposes. 
  return wav[audio_mask == True]