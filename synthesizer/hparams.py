#
# hparams.py
#
# All hyperparameters used in this component. Includes audio parameters
# as well as model parameters. 

import pprint

class HParams(object):
  # Acts like a dictionary. 
  def __init__(self, **kwargs): self.__dict__.update(kwargs)
  def __setitem__(self, key, value): setattr(self, key, value)
  def __getitem__(self, key): return getattr(self, key)
  def __repr__(self): return pprint.pformat(self.__dict__)

  
# Default hparams.

_hparams_tts_schedule_batch_size = 32 # Consistent batch size for all schedules. 

hparams = HParams(
  # Signal Processing (Used in both synthesizer and vocoder)
  sample_rate = 16000,
  n_fft = 800,
  num_mels = 80,
  hop_size = 200, # Tacotron uses 12.5 ms frame shift (sample_rate * 0.0125)
  win_size = 800, # Tacotron uses 50ms frame length (sample_rate * 0.050)
  fmin = 55,
  min_level_db = -100,
  ref_level_db = 20,
  max_abs_value = 4.,
  preemphasis = 0.97,
  preemphasize = True,

  # Tacotron Text to Speech - Model Architecture
  tts_embed_dims = 512,
  tts_encoder_dims = 256,
  tts_decoder_dims = 128,
  tts_postnet_dims = 512,
  tts_encoder_K = 5,
  tts_lstm_dims = 1024,
  tts_postnet_K = 5,
  tts_num_highways = 4,
  tts_dropout = 0.5,
  tts_cleaner_names = ["english_cleaners"],
  # Value below which audio generation ends. 
  # Ex) range of [-4,4], this will terminate the sequence at the first
  # frame that has all values < -3.4.
  tts_stop_threshold = -3.4,

  tts_dataloader_num_workers = 4, # Added for performance improvements.

  # Tacotron Training - Hyperparameters
  tts_schedule = [(2,  1e-3,  20_000,  _hparams_tts_schedule_batch_size),   # Progressive training schedule
                  (2,  5e-4,  40_000,  _hparams_tts_schedule_batch_size),   # (r, lr, step, batch_size)
                  (2,  2e-4,  80_000,  _hparams_tts_schedule_batch_size),   #
                  (2,  1e-4, 160_000,  _hparams_tts_schedule_batch_size),   # r = reduction factor (# of mel frames
                  (2,  3e-5, 320_000,  _hparams_tts_schedule_batch_size),   #     synthesized for each decoder iteration)
                  (2,  1e-5, 640_000,  _hparams_tts_schedule_batch_size)],  # lr = learning rate

  tts_clip_grad_norm = 1.0, # Prevents gradient explosion (set to None if not desired)
  tts_eval_interval = 500,  # Num steps between evaluation. Set to -1 for every epoch, or 0 to disable. 
  tts_eval_num_samples = 1, # Number of samples during evaluation. 

  # Data preprocessing
  #max_mel_frames = 1800,
  max_mel_frames = 900,
  rescale = True,
  rescaling_max = 0.9,
  synthesis_batch_size = 16, # For vocoder preprocessing and inference.

  # Mel Visualization + Griffin-Lim
  signal_normalization = True,
  power = 1.5,
  griffin_lim_iters = 60,

  # Audio preprocessing options
  fmax = 7600, # Should not exceed (sample_rate // 2)
  allow_clipping_in_normalization = True, # Used when signal_normalization is enabled.
  clip_mels_length = True, # When enabled, discards samples exceeding max_mel_frames.
  use_lws = False, # Fast spectogram phase recovery using local weighted sums.
  symmetric_mels = True, # Sets mel range to [-max_abs_value, max_abs_value] if true. [0, max_abs_value] otherwise.
  trim_silence = True, # Use with sample rate of 16000 for best results.

  # SV2TTS Stuff.
  speaker_embedding_size = 256, # Dimension of the speaker encoder model you're using.
  silence_min_duration_split = 0.4, # Duration in seconds of silence for an utterance to be split.
  utterance_min_duration = 1.6, # Duration in seconds below which utterances are discarded. 
  )

# For debug purposes, vomit everything out. 
def hparams_debug_string():
  return str(hparams)