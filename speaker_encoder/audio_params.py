#
# audio_params.py
#
# Various configurable constants used by audio file and content
# manipulations. These should stay the same, but can be tweaked
# to test performance (Ex) inference_n_frames 80 vs 160; the former
# is what the authors of the GE2E paper advocated).

# Log Mel Spectograms - input to encoder model
mel_window_length = 25 # Milliseconds - window length of input mel spectogram
mel_window_step = 10 # Milliseconds - step of input mel spectogram
mel_n_channels = 40 # 40-channel mel spectograms

# Audio Properties
sampling_rate = 16000 # Milliseconds - all samples in our 3 datasets are sampled at 16kHz
partials_n_frames = 160 # 1600 ms - Number of spectogram frames each partial utterance
inference_n_frames = 80 # 800 ms - Number of spectogram frames used at inference time

# Preprocessing - Voice Activation Detection
vad_window_length = 30 # Milliseconds (10, 20, or 30) - Granularity of the VAD operation
vad_moving_average_width = 8 # Num Frames averaged together during VAD. Larger = less smooth
vad_max_silence_length = 6 # Max acceptable silent frames a segment is allowed

# Preprocessing - Audio Volume Normalization
audio_norm_target_dBFS = -30 # Targeted volume