#
# sv2tts_inference.py
#
# Bridge between CorentinJ's pretrained vocoder and our modified
# multispeaker synthesis project for inference. The code in the
# SV2TTS folder has been modified accordingly. 

from vocoder.SV2TTS.inference import *
from vocoder.SV2TTS.hparams import *

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

class SV2TTSBridge: 
  _weights_fpath = "./SV2TTS/pretrained/vocoder.pt"
  _num_workers = 10
  _target = 4000
  _overlap = 150
  # How many samples of 0s of silence for pauses between mels. 
  _silence_spaces = 10

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
    separated_mels = []
    min_mel_value = 9999
    for mel in mels:
      min_mel_value = min(min_mel_value, np.min(mel))

    for i in range(0, len(mels)):
      mel = mels[i]
      separated_mels.append(mel)
      #print(mel)
      #input()
      if i < len(mels):
        separated_mels.append(np.full(shape=(mel.shape[0], self._silence_spaces), fill_value=min_mel_value))
    combined_mels = np.hstack(separated_mels)
    wavs.append(infer_waveform(combined_mels, target=self._target, overlap=self._overlap))

    return wavs

  def sv2tts_worker(self, mel):
    load_model(self._weights_fpath)
    return infer_waveform(mel)

