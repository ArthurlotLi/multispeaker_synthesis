#
# inference.py
#
# Usage of pretrained synthesizer models. Loads the specified 
# tacotron model and works on provided text + embedding(s).

import synthesizer.audio_utils as audio
from synthesizer.hparams import hparams
from synthesizer.tacotron import Tacotron
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import text_to_sequence

import torch
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa

# External class to be used during runtime. 
class Synthesizer:
  sample_rate = hparams.sample_rate
  hparams = hparams

  # Note: The model itself is not loaded into memory until it's used
  # OR load is explicitly called, OR if the load_immediately flag is
  # set.
  def __init__(self, model_fpath: Path, verbose=True, load_immediately=False):
    self.model_fpath = model_fpath
    self.verbose = verbose

    # Check for GPU and use it, else CPU.
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    if self.verbose:
      print("[INFO] Synthesizer Inference - Using device: ", self.device)
    
    self._model = None
    if load_immediately:
      self.load()
  
  # Have we loaded it yet? 
  def is_loaded(self):
    return self._model is not None

  # Instantiates and loads the model given the provided model. 
  def load(self):
    if self.verbose:
      print("[INFO] Synthesizer Inference - Loading model from: %s" % self.model_fpath)
    self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
                           num_chars=len(symbols),
                           encoder_dims=hparams.tts_encoder_dims,
                           decoder_dims=hparams.tts_decoder_dims,
                           n_mels=hparams.num_mels,
                           fft_bins=hparams.num_mels,
                           postnet_dims=hparams.tts_postnet_dims,
                           encoder_K=hparams.tts_encoder_K,
                           lstm_dims=hparams.tts_lstm_dims,
                           postnet_K=hparams.tts_postnet_K,
                           num_highways=hparams.tts_num_highways,
                           dropout=hparams.tts_dropout,
                           stop_threshold=hparams.tts_stop_threshold,
                           speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)
    
    self._model.load(self.model_fpath)

    # Always set the model to evaluation mode. 
    self._model.eval()

    if self.verbose:
      print("[INFO] Synthesizer Inference - Loaded synthesizer \"%s\" trained to step %d" % (self.model_fpath.name, self._model.state_dict()["step"]))

  # Principal function - provided texts, embeddings and optionally 
  # whether to return a matrix of alignments between characters, 
  # execute model inference to generate output spectogram(s). 
  def synthesize_spectograms(self, texts: List[str], 
                             embeddings: Union[np.ndarray, 
                             List[np.ndarray]], return_alignments = False):
    if not self.is_loaded():
      self.load()
    
    # Preprocess the text inputs. 
    if self.verbose:
      print("[DEBUG] Synthesizer Inference - Preprocessing input text + embeddings.")
    inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
    if not isinstance(embeddings, list):
      embeddings = [embeddings]
    
    # Batch input text + embeddings.
    batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                      for i in range(0, len(inputs), hparams.synthesis_batch_size)]
    batched_embeds = [embeddings[i:i+hparams.synthesis_batch_size] 
                      for i in range(0, len(embeddings), hparams.synthesis_batch_size)]
    
    # Execute synthesis. 
    specs = []
    for i, batch in enumerate(batched_inputs, 1):
      if self.verbose:
        print("[DEBUG] Synthesizer Inference - Processing %d out of %d." % (i, len(batched_inputs)))
      
      # Pad texts so they are all of the same length.
      text_lens = [len(text) for text in batch]
      max_text_len = max(text_lens)
      chars = [self.pad1d(text, max_text_len) for text in batch]
      chars = np.stack(chars)

      # Stack speaker embeddings into 2D array so we can process 
      # them in a batch. 
      speaker_embeds = np.stack(batched_embeds[i-1])

      # Convert into tensors. 
      chars = torch.tensor(chars).long().to(self.device)
      speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

      # Inference
      _, mels, alignments = self._model.generate(chars, speaker_embeddings)
      mels = mels.detach().cpu().numpy()

      # Post process the mels real quick. 
      for m in mels:
        # Trim silence from the end of each spectogram. 
        while np.max(m[:, -1]) < hparams.tts_stop_threshold:
          m = m[:, :-1]
        specs.append(m)
    
    # All done. 
    return(specs, alignments) if return_alignments else specs
  
  # Loads and preprocesses an audio file identically to how training
  # samples were preprocessed.
  @staticmethod
  def load_preprocess_wav(fpath):
    wav = librosa.load(str(fpath), hparams.sample_rate)[0]
    if hparams.rescale:
      wav = wav / np.abs(wav).max() * hparams.rescaling_max
    return wav
  
  # Creates a mel spectogram from an audio file in the same manner
  # as how the solution mel spectograms were preprocessed during
  # training. 
  @staticmethod
  def make_spectogram(fpath_or_wav: Union[str, Path, np.ndarray]):
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
      wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
    else:
      wav = fpath_or_wav
    
    # Generate the spectogram. 
    mel_spectogram = audio.melspectogram(wav, hparams).astype(np.float32)
    return mel_spectogram
  
  # Inverts a mel spectogram using Griffin-Lim - expects the input
  # mel to have been built with identical parameters to hparams.py
  @staticmethod
  def griffin_lim(mel):
    return audio.inv_mel_spectogram(mel, hparams)

  @staticmethod
  # Helper function. 
  def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)