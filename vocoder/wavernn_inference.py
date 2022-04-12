#
# wavernn_inference.py
#
# Bridges between the prebuilt WaveRNN git repository + pretrained
# WaveRNN vocoder and our multispeaker synthesis implementation. 
# Allows for production inference with their vocoder. 
#
# We're using a third-party vocoder as this point is the most crutial
# when it comes to viability in a production environment.

import sys
sys.path.append("./vocoder/WaveRNN")
from utils.dataset import get_vocoder_datasets
from utils.dsp import *
from models.fatchord_version import WaveRNN
from utils.paths import Paths
from utils.display import simple_table

import torch
from pathlib import Path
import numpy as np

class WaveRNNBridge:
  _hp_file = "./vocoder/WaveRNN/hparams.py"

  def __init__(self, force_cpu = False, load_immediately=True):
    self.force_cpu = force_cpu
    if load_immediately is True:
      self._load_wavernn()
    
  def _load_wavernn(self):
    hp.configure(self._hp_file)  # Load hparams from file
    # set defaults for any arguments that depend on hparams

    if not self.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('[INFO] WaveRNN - Using device:', device)

    print('[INFO] WaveRNN - Initialising Model...')

    self._model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).to(device)
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    voc_weights = paths.voc_latest_weights
    self._model.load(voc_weights)

    print('[INFO] WaveRNN - Pretrained model initialized successfully!')

  def wavernn(self, mels):
    k = self._model.get_step() // 1000
    target = hp.voc_target
    batched = hp.voc_gen_batched
    overlap = hp.voc_overlap
    save_path = "TEMP_FILE.wav"

    for mel in mels:
      if mel.ndim != 2 or mel.shape[0] != hp.num_mels:
          raise ValueError(f'Expected a numpy array shaped (n_mels, n_hops), but got {wav.shape}!')

      # Fix mel spectrogram scaling to be from 0 to 1
      mel = (mel + 4) / 8
      np.clip(mel, 0, 1, out=mel)

      #_max = np.max(mel)
      #_min = np.min(mel)
      #if _max >= 1.01 or _min <= -0.01:
          #raise ValueError(f'Expected spectrogram range in [0,1] but was instead [{_min}, {_max}]')

      mel = torch.tensor(mel).unsqueeze(0)

      batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
      save_str = save_path

      _ = self._model.generate(mel, save_str, batched, target, overlap, hp.mu_law)

    return None