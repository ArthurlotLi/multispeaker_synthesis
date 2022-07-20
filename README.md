# Multispeaker Text To Speech Synthesis

Informed by state-of-the-art research into Multispeaker Text to Speech Synthesis [1]. A neural network-based system generating a waveform file of speech given a few seconds of untranscribed reference audio from a target speaker as well as the output text. 

Guided by code published by [2]. Features a wide variety of modifications and enhancements in terms of performance and usability, including implementations of model evaluation and visualization that were otherwise absent.

Models augmented with data from Tales Skits dataset generation algorithms and Tales of Skits web application. 

For more information, please see [Tales of Skits](http://talesofskits.com/).

[![Tales of Skits Website](https://i.imgur.com/9HlmT9X.png "Tales of Skits Website")](http://talesofskits.com/)

Contains three independently trained components:

1. Speaker Encoder - Provides a representation of speaker qualities having been trained on a separate Speaker Verification task. This fixed-dimensional vector is passed to (2).

2. Synthesis - Based on Tacotron 2; generates a log mel spectogram from input text conditioned by embedded speaker information provided by (1).

3. Vocoder - Given the log mel spectogram output by (2), generates the output waveform file. Auto-regressive WaveNet-based. 

Datasets utilized:

VoxCeleb (1 and 2): https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ 

LibriSpeech: https://www.openslr.org/12/

VCTK: https://datashare.ed.ac.uk/handle/10283/2950

Tales Skits Dataset

[![Tales of Skits Website](https://i.imgur.com/A7HdMCQ.png "Tales of Skits Website")](http://talesofskits.com/)

---

### References:

[1] Jia, Ye, et al. "Transfer learning from speaker verification to multispeaker text-to-speech synthesis." Advances in neural information processing systems 31 (2018). https://arxiv.org/abs/1806.04558 

[2] Jemine, C. "Master thesis : Real-Time Voice Cloning. (Unpublished master's thesis)." (2019). Université de Liège, Liège, Belgique. https://matheo.uliege.be/handle/2268.2/6801

---

### Usage:

To use MultispeakerSynthesis, you need to make sure you do some manual steps to allow your local environment to function correctly.

1. Install pyaudio dependencies. Run:

   pip install pipwin
   pipwin install pyaudio

2. Install everything required by MultispeakerSynthesis. Run:

   pip install -r requirements.txt 

3. Install FFmpeg. The pip installation won't set the cli tool, so you
   need to install from here:

   https://ffmpeg.org/

   Make sure to add the directory with ffmpeg.exe to the system PATH,
   then restart your command line/terminal session.