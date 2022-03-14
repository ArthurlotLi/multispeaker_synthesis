#
# synthesizer_dataset.py
#
# Dataset object for loading preprocessed synthesizer data.

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence