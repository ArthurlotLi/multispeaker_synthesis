#
# speaker_batch.py
#
# Split a batch of speaker info into batches of utterances.

import numpy as np
from typing import List
from speaker_encoder.data_objects.speaker import Speaker
from speaker_encoder.data_objects.speaker import SpeakerSequential

class SpeakerBatch:
  def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
    self.speakers = speakers
    self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}

    # Tensor of shape: (n_speakers*n_utterances, n_frames, mel_n).
    # Ex) 3 speakers, 4 utterances each of 160 frames of 40 mel
    # coefficients: (12, 160, 40)
    self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

# Variant that allows for repeatable testing. 
class SpeakerBatchSequential:
  def __init__(self, speakers: List[SpeakerSequential], utterances_per_speaker: int, n_frames: int):
    self.speakers = speakers
    self.partials = {s: s.sequential_partial(utterances_per_speaker, n_frames) for s in speakers}

    # Tensor of shape: (n_speakers*n_utterances, n_frames, mel_n).
    # Ex) 3 speakers, 4 utterances each of 160 frames of 40 mel
    # coefficients: (12, 160, 40)
    self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])