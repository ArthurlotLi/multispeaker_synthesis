#
# production_inference.py
#
# Exposes full multispeaker synthesis functionality for external
# users (such as Kotakee Companion). Encodes, Synthesizes, and Vocodes
# given wav file(s) and input text and the appropriate model fpaths. 

from synthesizer.inference import *
#from vocoder.wavernn_inference import WaveRNNBridge
from vocoder.sv2tts_inference import SV2TTSBridge

import time
import re

class MultispeakerSynthesis:
  synthesizer = None
  embedding = None

  # If we get a sample that is longer than the max chars OR we get
  # a greater number than max samples, we'll be splitting into small
  # minibatches of the specified size so we don't run into VRAM or
  # RAM issues. 
  longest_max_chars = 300

  # Number of processes for griffin lim parallel processing. 
  griffin_lim_num_processes = 1

  # Expects fpaths for the synthesizer and optionally the encoder.
  # If the encoder path is provided and load_immediately is true,
  # then we will load the encoder as well. 
  def __init__(self, synthesizer_fpath, speaker_encoder_fpath = None,
               verbose=True, load_immediately=True, 
               target= None, overlap = None, batched = None):
    self.speaker_encoder_fpath = speaker_encoder_fpath
    self.synthesizer_fpath = synthesizer_fpath
    self.verbose = verbose

    self.synthesizer = Synthesizer(model_fpath = synthesizer_fpath, 
                               verbose=verbose,
                               load_immediately=load_immediately)
    
    #self.vocoder = WaveRNNBridge(load_immediately=load_immediately)
    self.sv2tts = SV2TTSBridge(load_immediately=load_immediately, 
      target=target, overlap=overlap, batched=batched)

    if speaker_encoder_fpath is not None and load_immediately:
      self._load_embedding()

  # Only load the encoder stuff if we need to use it.
  def _load_embedding(self):
    if self.speaker_encoder_fpath is not None:
      from production_embedding import Embedding
      self.embedding = Embedding(self.speaker_encoder_fpath)
  
  # Synthesizes audio given text + location of wav file to load.
  # Optionally may use Griffin Lim as the vocoder.  
  def synthesize_audio_from_audio(self, texts: List[str], 
                                  wav_fpath: Path,
                                  vocoder = "griffinlim"):
    if self.embedding is None:
      self._load_embedding()

    # Append texts into minibatches with a maximum char limit. 
    batches = []
    minibatch = []
    batch_chars = 0
    total_chars = 0
    for text in texts:
      text_len = len(text)
      total_chars += text_len
      new_batch_chars = batch_chars + text_len
      if new_batch_chars > self.longest_max_chars:
        batches.append(minibatch)
        minibatch = [text]
        batch_chars = text_len
      else:
        minibatch.append(text)
        batch_chars += text_len
    
    if len(minibatch) > 0:
      batches.append(minibatch)

    print("[DEBUG] Multispeaker Synthesis - Total chars: %d. Processing in %d batches." % (total_chars, len(batches)))

    start_time = time.time()

    total_mels = []
    for batch in batches:
      # Generate the encoding for the audio. 
      embeds = self.embedding.single_embedding(wav_fpath)

      # Duplicate the embeddings - one for each utterance.
      num_texts = len(batch)
      concat_embeds = []
      for i in range(num_texts): concat_embeds.append(embeds)
      mels = self.synthesizer.synthesize_spectograms(texts=batch, embeddings=concat_embeds)
      total_mels += mels

    print("[DEBUG] Multispeaker Synthesis - Synthesis completed %.4f seconds." % (time.time() - start_time))

    return self.vocode_mels(total_mels, vocoder)


  # Given texts to generate mels from as well as the fpath to the
  # speaker embedding, loads the embedding and generates a mel 
  # spectogram. 
  def synthesize_audio_from_embeds(self, texts: List[str],
                                   embeds_fpath: Path, 
                                   vocoder = "griffinlim"):
    embeds = np.load(embeds_fpath)

    # Append texts into minibatches with a maximum char limit. 
    batches = []
    minibatch = []
    batch_chars = 0
    total_chars = 0
    for text in texts:
      text_len = len(text)
      if text_len > 0:
        new_batch_chars = batch_chars + text_len
        if new_batch_chars > self.longest_max_chars:
          batches.append(minibatch)
          minibatch = [text]
          batch_chars = text_len
        else:
          minibatch.append(text)
          batch_chars += text_len
        total_chars += text_len
    
    if len(minibatch) > 0:
      batches.append(minibatch)

    print("[DEBUG] Multispeaker Synthesis - Total chars: %d. Processing in %d batches." % (batch_chars, len(batches)))

    start_time = time.time()

    total_mels = []
    for batch in batches:
      # Duplicate the embeddings - one for each utterance.
      num_texts = len(batch)
      concat_embeds = []
      for i in range(num_texts): concat_embeds.append(embeds)
      mels = self.synthesizer.synthesize_spectograms(texts=batch, embeddings=concat_embeds)
      total_mels += mels

    print("[DEBUG] Multispeaker Synthesis - Synthesis completed %.4f seconds." % (time.time() - start_time))

    return self.vocode_mels(total_mels, vocoder = vocoder)

  def vocode_mels(self, mels, vocoder):
    """
    Given mels, provides audio wav. This can be done with either the
    Griffin Lim vocoder or the WaveRNN vocoder, specified by a string
    argument. 
    """
    start_time = time.time()
    wavs = []
    #if vocoder == "wavernn":
      #print("[DEBUG] Multispeaker Synthesis - Submitting mel spectrograms to WaveRNN.")
      #wavs = self.vocoder.wavernn(mels)
      #print("[DEBUG] Multispeaker Synthesis - WaveRNN completed %.4f seconds." % (time.time() - start_time))
    if vocoder == "sv2tts":
      print("[DEBUG] Multispeaker Synthesis - Submitting mel spectrograms to SV2TTS Vocoder.")
      wavs = self.sv2tts.sv2tts(mels)
      print("[DEBUG] Multispeaker Synthesis - SV2TTS Vocoder completed %.4f seconds." % (time.time() - start_time))
    elif vocoder == "griffinlim":
      print("[DEBUG] Multispeaker Synthesis - Submitting mel spectrograms to Griffin Lim.")
      for mel in mels:
        wavs.append(audio.inv_mel_spectogram(mel, hparams))
      print("[DEBUG] Multispeaker Synthesis - Griffin Lim completed %.4f seconds." % (time.time() - start_time))
    else:
      assert False # No good very bad day.
    return wavs
  
  # Expose this for users. 
  def save_wav(self, wav, path):
    audio.save_wav(wav, path, sr=hparams.sample_rate)


# Debug only. Example usage:
# python production_inference.py "How are you today?" ELEANOR
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("text_to_speak")
  parser.add_argument("speaker")
  args = parser.parse_args()

  model_num = "model6"
  #model_num_synthesizer = "model6_edna"
  #model_num_synthesizer = "model6_velvet"
  #model_num_synthesizer = "model6_eleanor"
  model_num_synthesizer = "model6"

  vocoder = "sv2tts"
  #vocoder = "griffinlim"

  synthesizer_fpath = Path("./production_models/synthesizer/"+model_num_synthesizer+"/synthesizer.pt")
  speaker_encoder_fpath = Path("./production_models/speaker_encoder/"+model_num+"/encoder.pt")

  #embeds_fpath = Path("../kotakee_companion/assets_audio/multispeaker_synthesis_speakers/"+ args.speaker+ ".npy")
  embeds_fpath = Path("../multispeaker_synthesis_speakers/"+ args.speaker+ ".npy")
  #embeds_fpath = Path("../multispeaker_synthesis_speakers_verified/test/"+ args.speaker+ "/"+args.emotion + ".npy")

  text_to_speak = [args.text_to_speak]
  split_sentence_re = r'[\.|!|,|\?|:|;|-] '

  # The only preprocessing we do here is to split sentences into
  # different strings. This makes the pronunciation cleaner and also
  # makes inference faster (as it happens in a batch).
  processed_texts = []
  for text in text_to_speak:
    split_text = re.split(split_sentence_re, text)
    processed_texts += split_text
    processed_texts

  debug_out = "./debug_%s.wav" % model_num

  multispeaker_synthesis = MultispeakerSynthesis(synthesizer_fpath=synthesizer_fpath,
                                                 speaker_encoder_fpath=speaker_encoder_fpath)
  wavs = multispeaker_synthesis.synthesize_audio_from_embeds(texts = processed_texts, embeds_fpath = embeds_fpath, 
                                                             vocoder = vocoder)

  def play_wavs(wavs, debug_out):
    for wav in wavs:
      audio.save_wav(wav, debug_out, sr=hparams.sample_rate)

      from pydub import AudioSegment

      # Normalize the audio. Not the best code, but it works in ~0.007 seconds.
      wav_suffix = debug_out.rsplit(".", 1)[1]
      sound = AudioSegment.from_file(debug_out, wav_suffix)
      change_in_dBFS = -15.0 - sound.dBFS
      normalized_sound = sound.apply_gain(change_in_dBFS)
      normalized_sound.export(debug_out, format=wav_suffix)

      import wave
      import pyaudio  

      chunk = 1024
      f = wave.open(debug_out, "rb")
      p = pyaudio.PyAudio()
      stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                  channels = f.getnchannels(),  
                  rate = f.getframerate(),  
                  output = True) 
      data = f.readframes(chunk)
      while data:  
        stream.write(data)  
        data = f.readframes(chunk)
      stream.stop_stream()  
      stream.close()  

      #close PyAudio  
      p.terminate()
  
  play_wavs(wavs, debug_out)

  import os
  if os.path.exists(debug_out):
    os.remove(debug_out)
