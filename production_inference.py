#
# production_inference.py
#
# Exposes full multispeaker synthesis functionality for external
# users (such as Kotakee Companion). Encodes, Synthesizes, and Vocodes
# given wav file(s) and input text and the appropriate model fpaths. 

from synthesizer.inference import *

import time

class MultispeakerSynthesis:
  synthesizer = None
  embedding = None

  # If we get a sample that is longer than the max chars OR we get
  # a greater number than max samples, we'll be splitting into small
  # minibatches of the specified size so we don't run into VRAM or
  # RAM issues. 
  longest_max_chars = 200

  # Expects fpaths for the synthesizer and optionally the encoder.
  # If the encoder path is provided and load_immediately is true,
  # then we will load the encoder as well. 
  def __init__(self, synthesizer_fpath, speaker_encoder_fpath = None,
               verbose=True, load_immediately=True):
    self.speaker_encoder_fpath = speaker_encoder_fpath
    self.synthesizer_fpath = synthesizer_fpath
    self.verbose = verbose

    self.synthesizer = Synthesizer(model_fpath = synthesizer_fpath, 
                               verbose=verbose,
                               load_immediately=load_immediately)
    
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
                                  wav_fpath: Path):
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

    return self.vocode_mels(total_mels)


  # Given texts to generate mels from as well as the fpath to the
  # speaker embedding, loads the embedding and generates a mel 
  # spectogram. 
  def synthesize_audio_from_embeds(self, texts: List[str],
                                   embeds_fpath: Path):
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

    total_mels = []
    for batch in batches:
      # Duplicate the embeddings - one for each utterance.
      num_texts = len(batch)
      concat_embeds = []
      for i in range(num_texts): concat_embeds.append(embeds)
      mels = self.synthesizer.synthesize_spectograms(texts=batch, embeddings=concat_embeds)
      total_mels += mels

    return self.vocode_mels(total_mels)

  # Given mels, provides audio wav. 
  def vocode_mels(self, mels):
    # TODO: for now, griffin lim is hard coded. 
    print("[DEBUG] Multispeaker Synthesis - Submitting mel spectrograms to Griffin Lim.")
    start_time = time.time()
    wavs = []
    for mel in mels:
      wavs.append(audio.inv_mel_spectogram(mel, hparams))
    print("[DEBUG] Multispeaker Synthesis - Griffin Lim completed %.4f seconds." % (time.time() - start_time))
    return wavs
  
  # Expose this for users. 
  def save_wav(self, wav, path):
    audio.save_wav(wav, path, sr=hparams.sample_rate)


# Debug only. Example usage:
# python production_inference.py pretrained
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("model_num")
  args = parser.parse_args()

  synthesizer_fpath = Path("./production_models/synthesizer/"+args.model_num+"/synthesizer.pt")
  speaker_encoder_fpath = Path("./production_models/speaker_encoder/"+args.model_num+"/encoder.pt")

  wav_fpath = Path("../multispeaker_synthesis_speakers/eleanor/Neutral.wav")
  text_to_speak = ["The weather outside is sunny", "clear skies", "with a maximum of 85 and a minimum of 75","Humidity is 15 percent."]

  debug_out = "./debug_" + args.model_num+ ".wav"

  multispeaker_synthesis = MultispeakerSynthesis(synthesizer_fpath=synthesizer_fpath,
                                                 speaker_encoder_fpath=speaker_encoder_fpath)
  wavs = multispeaker_synthesis.synthesize_audio_from_audio(texts = text_to_speak, wav_fpath = wav_fpath)

  for wav in wavs:
    audio.save_wav(wav, debug_out, sr=hparams.sample_rate)

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
