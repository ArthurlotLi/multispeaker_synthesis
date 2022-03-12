#
# speaker.py
#
# Contains the set of utterances of a single speaker + methods
# to access/sample them. 

from speaker_encoder.data_objects.random_cycler import RandomCycler
from speaker_encoder.data_objects.sequential_cycler import SequentialCycler
from speaker_encoder.data_objects.utterance import Utterance
from pathlib import Path

class Speaker:
  # Expects the root path of that speaker.
  def __init__(self, root: Path):
    self.root = root
    self.name = root.name
    self.utterances = None
    self.utterance_cycler = None
  
  # Load the stuff from the fpaths contained inside of Utterance. 
  # Create a random cycler as well. 
  def _load_utterances(self):
    with self.root.joinpath("_sources.txt").open("r") as sources_file:
      sources = [l.split(",") for l in sources_file]
    sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
    self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
    self.utterance_cycler = RandomCycler(self.utterances)

  # Samples a batch of a certain number of unique, partial utterances
  # in such a way that all utterances come up at least once every 
  # two cycles and in a random order every time. 
  #
  # Expects the number of partial utterances to sample from the speaker.
  # Note that utterances are guaranteed not to be repeated if count is
  # not larger than the number of utterances available. ALso expects
  # the number of frames in a parital utterance.
  #
  # Returns a list of tuples: (utterance, frames, range), where
  # utterance is an Utterance, frames are the frames of the partial
  # utterance and the range is the range of the partial utterance 
  # relative to the complete utterance. 
  def random_partial(self, count, n_frames):
    if self.utterances is None:
      self._load_utterances()
    
    utterances = self.utterance_cycler.sample(count)

    a = [(u,) + u.random_partial(n_frames) for u in utterances]

    return a

# Allows for repeatable testing. 

class SpeakerSequential:
  # Expects the root path of that speaker.
  def __init__(self, root: Path):
    self.root = root
    self.name = root.name
    self.utterances = None
    self.utterance_cycler = None
  
  # Load the stuff from the fpaths contained inside of Utterance. 
  # Create a random cycler as well. 
  def _load_utterances(self):
    with self.root.joinpath("_sources.txt").open("r") as sources_file:
      sources = [l.split(",") for l in sources_file]
    sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
    self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
    self.utterance_cycler = SequentialCycler(self.utterances)

  # Returns a list of tuples: (utterance, frames, range), where
  # utterance is an Utterance, frames are the frames of the partial
  # utterance and the range is the range of the partial utterance 
  # relative to the complete utterance. 
  def sequential_partial(self, count, n_frames):
    if self.utterances is None:
      self._load_utterances()
    
    utterances = self.utterance_cycler.sample(count)

    a = [(u,) + u.sequential_partial(n_frames) for u in utterances]

    return a