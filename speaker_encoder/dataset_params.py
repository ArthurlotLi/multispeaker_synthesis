#
# dataset_params.py
#
# Constants indicating the file structure of our datasets, wherever
# they are. Also specifies, when preprocessing, what nationalities 
# to include with voxceleb1 (voxceleb2 doesn't include this info)

# Nationalities for English speakers... or at least mostly.
anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]

# Librispeech. We're using other for speaker encoder, clean for
# synthesizer.
librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}

# Voxceleb stuff. We're using these exclusively for the speaker
# encoder. 
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

# Libritts - a multispeaker speaker corpus derived from the
# original Librispeech dataset. 24kHZ sampling rate, speech
# is split at sentence breaks, both original and normalized
# texts included, contextual info can be extracted, and
# utterances with significant background noise are excluded. 
#
# NOTE: for now, this is unused. 
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}

# NOTE: For now, these are unused. 
other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]