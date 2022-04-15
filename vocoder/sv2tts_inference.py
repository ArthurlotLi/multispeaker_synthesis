#
# sv2tts_inference.py
#
# Bridge between CorentinJ's pretrained vocoder and our modified
# multispeaker synthesis project for inference. The code in the
# SV2TTS folder has been modified accordingly. 

from vocoder.SV2TTS.inference import *
from vocoder.SV2TTS.hparams import *
from vocoder.SV2TTS.audio import *
from speaker_encoder.audio_utils import normalize_volume
from speaker_encoder.audio_params import *

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

class SV2TTSBridge: 
  _weights_fpath = "../production_models/vocoder/model6/vocoder.pt"
  #_weights_fpath = "./SV2TTS/pretrained/vocoder.pt"
  _num_workers = 10
  _target = 2000
  _overlap = 100
  # How many samples of 0s of silence for pauses between mels. 
  _silence_spaces = 10
  _add_silences = False

  def __init__(self, load_immediately=True):
    self._weights_fpath = str(Path(__file__).parent.resolve().joinpath(self._weights_fpath))
    if load_immediately:
      load_model(self._weights_fpath)
  
  def sv2tts(self,mels):
    wavs = []
    # Naive, slow method. 127 seconds for our test statement.
    #for mel in mels:
      #wavs.append(infer_waveform(mel))
    
    # Better. 67 seconds for our test statement.
    #job = Pool(self._num_workers).imap(self.sv2tts_worker, mels)
    #wavs = list(tqdm(job, "SV2TTS Vocoding", len(mels), unit="wavs"))

    # Combine all the mels by appending them, allowing for far better batch work.
    normed_mels = []
    for mel in mels:
      normed_mels.append(mel / hp.mel_max_abs_value)

    separated_mels = []
    min_mel_value = 9999
    for mel in normed_mels:
      min_mel_value = min(min_mel_value, np.min(mel))

    for i in range(0, len(normed_mels)):
      mel = normed_mels[i]
      separated_mels.append(mel)
      #print(mel)
      #input()
      if self._add_silences and i < len(normed_mels):
        separated_mels.append(np.full(shape=(mel.shape[0], self._silence_spaces), fill_value=min_mel_value))
    combined_mels = np.hstack(separated_mels)

    #combined_mels = np.concatenate(mels, axis=1)
    waveform = infer_waveform(combined_mels, target=self._target, overlap=self._overlap, normalize = False)

    # Normalize the loudness of the result to offset some model 
    # wonkiness. 
    normalize_volume(waveform, audio_norm_target_dBFS)
    
    wavs.append(waveform)

    return wavs

  def sv2tts_worker(self, mel):
    load_model(self._weights_fpath)
    return infer_waveform(mel)

