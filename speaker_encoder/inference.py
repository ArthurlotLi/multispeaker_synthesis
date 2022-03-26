#
# inference.py
#
# Usage of trained speaker encoder models. 

from functools import partial
from speaker_encoder.audio_params import *
from speaker_encoder.model import SpeakerEncoder
from speaker_encoder.audio_utils import preprocess_audio # This should be exposed for users from here. 
import speaker_encoder.audio_utils

from matplotlib import cm
from pathlib import Path
import numpy as np
import torch

_model = None # our SpeakerEncoder model. 
_device = None # Our torch device

# Loads model into memory - can either be called explicitly or run
# implicitly on the first call to embed_frames with the default 
# weights file. 
#
# Expects the path to saved model weights + torch device. Defaults
# to GPU if available, otherwise will use CPU. Outputs (loss_device),
# will always be on the CPU (since that almost always is faster)
def load_model(weights_fpath: Path, device=None):
  global _model, _device
  # If not given a device, load it. otherwise, use the given device.
  if device is None:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  elif isinstance(device, str):
    _device = torch.device(device) # Ex) "cpu", "gpu"

  # Load the model with the device.
  _model = SpeakerEncoder(_device, torch.device("cpu"))
  checkpoint = torch.load(weights_fpath, _device)
  _model.load_state_dict(checkpoint["model_state"], strict=False)

  # Always set the model into evaluation/inference mode.
  _model.eval()

  print("[INFO] Inference - Loaded speaker encoder \"%s\" model trained to step %d." % (weights_fpath.name, checkpoint["step"]))

# Check if it's loaded. 
def is_loaded():
  return _model is not None

# Given a batch of mel spectograms, computes their embeddings with
# the model. Expects the spectograms, duh.
def embed_frames_batch(frames_batch):
  if _model is None:
    raise Exception("Model was not loaded! Call load_model() before inference.")
  
  # Send the spectograms to be analyzed to the device and then 
  # propigate them through the network with forward prop.
  frames = torch.from_numpy(frames_batch).to(_device)
  embed = _model.forward(frames).detach().cpu().numpy()
  return embed

# Determines where to split an utterance waveform and it's corresponding
# mel spectogram to obtain partial utterances of the specified
# length (each). Both the waveform and the mel spectogram slices are
# returned, making each partial utterance waveform correspond to the
# split spectogram. 
#
# Assumes that the mel spectogram parameters are exactly like those 
# defined in audio_params. 
#
# Note the returned ranges may index further than the length of the 
# waveform to fit the entire thing - it's recommneded to pad the
# waveform with zeros up to wave_slices[-1]. 
def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames, 
                           min_pad_coverage=0.75, overlap=0.5):
  assert 0 <= overlap < 1
  assert 0 < min_pad_coverage <= 1

  samples_per_frame = int((sampling_rate * mel_window_step / 1000))
  n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
  frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

  # Compute the slices for both wav and mel (equal slices, that is).
  wav_slices, mel_slices = [], []
  steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
  for i in range(0, steps, frame_step):
    mel_range = np.array([i, i+ partial_utterance_n_frames])
    wav_range = mel_range * samples_per_frame
    mel_slices.append(slice(*mel_range))
    wav_slices.append(slice(*wav_range))
  
  # Evaluate if extra padding is necessary - if so, add it. 
  last_wav_range = wav_slices[-1]
  coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
  if coverage < min_pad_coverage and len(mel_slices) > 1:
    mel_slices = mel_slices[:-1]
    wav_slices = wav_slices[:-1]
  
  return wav_slices, mel_slices

def embed_utterances(wavs, using_partials=True, return_partials=False, **kwargs):
  """
  Works with an arbitrary number of wavs. Reads in all wavs into a
  giant wav file, computes slices, and then provides them to the 
  model. 
  """
  # Combine the wavs.
  wav = None
  for wav_sample in wavs:
    if wav is None: 
      wav = wav_sample
    else:
      wav = np.concatenate((wav, wav_sample))

  # Process the entire utternace if not using partials (easy to do)
  if not using_partials:
    frames = speaker_encoder.audio_utils.wav_to_mel_spectogram(wav)
    embed = embed_frames_batch(frames[None, ...])[0]
    if return_partials:
      return embed, None, None
    return embed
  
  # Otherwise, compute where to split the utterance + pad if necessary. 
  wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
  max_wave_length = wave_slices[-1].stop

  print("[INFO] Inference - Computed %d partial slices." % len(mel_slices))

  # If necessary, execute padding.
  if max_wave_length > len(wav):
    wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
  
  # Split the utternace into partials.
  frames = speaker_encoder.audio_utils.wav_to_mel_spectogram(wav)
  frames_batch = np.array([frames[s] for s in mel_slices])

  # Submit the frames to the model. 
  partial_embeds = embed_frames_batch(frames_batch)

  # Compute the utterance embedding from the partials. This is 
  # more "friendly" to the model, as it was trained this way. 
  raw_embed = np.mean(partial_embeds, axis=0)

  # Normalize it. 
  embed = raw_embed / np.linalg.norm(raw_embed, 2)

  print("[INFO] Inference - Final raw embed computed by averaging all %d partial embeds." % len(partial_embeds))

  if return_partials:
    return embed, partial_embeds, wave_slices
  return embed

# Computes an embedding given a single wav file. This means we'll
# need to split it up into different utterances if we're using
# partials. 
#
# Expects the wav + options. Returns the embedding as a npy array
# of float32, shape (model_embedding_size,). If returning partials,
# the partial utterances are returned as shape 
# (n_partials, model_embedding_size) + wav partials as a list of slices. 
def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
  # Process the entire utternace if not using partials (easy to do)
  if not using_partials:
    frames = speaker_encoder.audio_utils.wav_to_mel_spectogram(wav)
    embed = embed_frames_batch(frames[None, ...])[0]
    if return_partials:
      return embed, None, None
    return embed
  
  # Otherwise, compute where to split the utterance + pad if necessary. 
  wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
  max_wave_length = wave_slices[-1].stop

  # If necessary, execute padding.
  if max_wave_length > len(wav):
    wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
  
  # Split the utternace into partials.
  frames = speaker_encoder.audio_utils.wav_to_mel_spectogram(wav)
  frames_batch = np.array([frames[s] for s in mel_slices])

  # Submit the frames to the model. 
  partial_embeds = embed_frames_batch(frames_batch)

  # Compute the utterance embedding from the partials. This is 
  # more "friendly" to the model, as it was trained this way. 
  raw_embed = np.mean(partial_embeds, axis=0)

  # Normalize it. 
  embed = raw_embed / np.linalg.norm(raw_embed, 2)

  if return_partials:
    return embed, partial_embeds, wave_slices
  return embed

# Visualization. Here for convenience. 
def plot_embedding_as_heatmap(embed, ax=None, title="", shape=None, color_range=(0,0.30)):
  import matplotlib.pyplot as plt
  if ax is None:
    ax = plt.gca()
  
  if shape is None:
    height = int(np.sqrt(len(embed)))
    shape = (height, -1)

  # Reshape the embedding to fit the shape of the graph.
  embed = embed.reshape(shape)

  # Graph it.
  cmap = cm.get_cmap()
  mappable = ax.imshow(embed, cmap=cmap)
  cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
  sm = cm.ScalarMappable(cmap=cmap)
  sm.set_clim(*color_range)

  ax.set_xticks([]), ax.set_yticks([])
  ax.set_title(title)

  # And show it. 
  plt.show()