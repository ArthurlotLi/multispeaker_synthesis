#
# utterance.py
#
# Data object for a single utterance of a speaker. Holds filepaths
# for the frames + wave.

import numpy as np

class Utterance:
  def __init__(self, frames_fpath, wave_fpath):
    self.frames_fpath = frames_fpath
    self.wave_fpath = wave_fpath

  def get_frames(self):
    return np.load(self.frames_fpath)
  
  # Crops the frames into a partial utterance of a certain number of
  # frames. Requires the number of frames to crop to. Returns the
  # partial utterance frames + tuple indicate the start and end of the
  # partial utterance in the complete utterance.
  def random_partial(self, n_frames):
    frames = self.get_frames()
    if frames.shape[0] == n_frames:
      start = 0
    else:
      start = np.random.randint(0, frames.shape[0] - n_frames)
    end = start + n_frames
    return frames[start:end], (start, end)
  
  # For sequential applications, always start at 0. 
  def sequential_partial(self, n_frames):
    frames = self.get_frames()
    start = 0
    end = start + n_frames
    return frames[start:end], (start, end)