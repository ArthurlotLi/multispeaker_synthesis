#
# speaker_verification_dataset.py
#
# Objects used during training to manage speaker data. 

from speaker_encoder.data_objects.random_cycler import RandomCycler
from speaker_encoder.data_objects.speaker_batch import SpeakerBatch
from speaker_encoder.data_objects.speaker import Speaker

from speaker_encoder.audio_params import partials_n_frames

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# How we access the speaker objects. 
class SpeakerVerificationDataset(Dataset):
  # Expects the root datasets file and retains each dir as a speaker
  # object. 
  def __init__(self, datasets_root: Path):
    self.root = datasets_root
    speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
    if len(speaker_dirs) == 0:
      raise Exception("SpeakerVerificationDataset - No speakers found! "
                      "Make sure you are pointing to the directory "
                      "containing all preprocessed speaker directories. ")
    self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
    self.speaker_cycler = RandomCycler(self.speakers)
  
  # How large this object is.
  def __len__(self):
    return int(1e10)
  
  # Randomly cycle to grab the next speaker. 
  def __getitem__(self, index):
    return next(self.speaker_cycler)

  # Grabs data of the txt files in the speaker folder. 
  def get_logs(self):
    log_string = ""
    for log_fpath in self.root.glob("*.txt"):
      with log_fpath.open("r") as log_file:
        log_string += "".join(log_file.readlines())
    return log_string

# The speaker labels. 
class SpeakerVerificationDataLoader(DataLoader):
  def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
               batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
               worker_init_fn=None):
    self.utterances_per_speaker = utterances_per_speaker

    # Initialize the DataLoader. 
    super().__init__(
      dataset=dataset, 
      batch_size=speakers_per_batch, 
      shuffle=False, 
      sampler=sampler, 
      batch_sampler=batch_sampler, 
      num_workers=num_workers,
      collate_fn=self.collate, 
      pin_memory=pin_memory, 
      drop_last=False, 
      timeout=timeout, 
      worker_init_fn=worker_init_fn
    )
  
  # Combines a group of speakers into a batch object. 
  def collate(self, speakers):
    return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames)