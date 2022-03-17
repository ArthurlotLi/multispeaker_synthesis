#
# train.py
#
# Training harness for Synthesizer models. 

import synthesizer.audio_utils as audio
from synthesizer.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils.torch_utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from vocoder.display import * 
from utils.profiler import Profiler

from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# Helper function to reduce clutter.
def np_now(x: torch.Tensor):
  return x.detach().cpu().numpy()

def time_string():
  return datetime.now().strftime("%Y-%m-%d %H:%M")

# Principal training function. 
def train(run_id: str, syn_dir: Path, models_dir: Path, save_every: int, 
          backup_every: int, force_restart: bool, hparams):

  # Directory housekeeping for models, plots, and data.
  models_dir.mkdir(exist_ok=True)

  model_dir = models_dir.joinpath(run_id)
  plot_dir = model_dir.joinpath("plots")
  wav_dir = model_dir.joinpath("wavs")
  mel_output_dir = model_dir.joinpath("mel-spectograms")
  meta_folder = model_dir.joinpath("metas")
  
  model_dir.mkdir(exist_ok=True)
  plot_dir.mkdir(exist_ok=True)
  wav_dir.mkdir(exist_ok=True)
  mel_output_dir.mkdir(exist_ok=True)
  meta_folder.mkdir(exist_ok=True)

  weights_fpath = model_dir / f"synthesizer.pt"
  metadata_fpath = syn_dir.joinpath("train.txt")

  print("[INFO] Train - Checkpoint path: {}".format(weights_fpath))
  print("[INFO] Train - Loading training data from: {}".format(metadata_fpath))

  # Bookkeeping.
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)

  # Code from WaveRNN/train_tacotron.py for loading. 
  if torch.cuda.is_available():
    device = torch.device("cuda")
    for session in hparams.tts_schedule:
      _, _, _, batch_size = session
      if batch_size % torch.cuda.device_count() != 0:
        raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
  else:
    device = torch.device("cpu")
  print("[INFO] Train - Using device: ", device)

  # Instantiate the Tacotron Model. Provide the hyperparameters to 
  # the model architecture.
  print("[INFO] Train - Initializing Tacotron Model...")
  model = Tacotron(embed_dims = hparams.tts_embed_dims,
                   num_chars = len(symbols),
                   encoder_dims=hparams.tts_encoder_dims,
                   decoder_dims=hparams.tts_decoder_dims,
                   n_mels = hparams.num_mels,
                   fft_bins = hparams.num_mels,
                   postnet_dims=hparams.tts_postnet_dims,
                   encoder_K=hparams.tts_encoder_K,
                   lstm_dims=hparams.tts_lstm_dims,
                   postnet_K=hparams.tts_postnet_K,
                   num_highways=hparams.tts_num_highways,
                   dropout=hparams.tts_dropout,
                   stop_threshold=hparams.tts_stop_threshold,
                   speaker_embedding_size=hparams.speaker_embedding_size).to(device)
  
  # Optimizer initialization. Adam, as always.
  optimizer = optim.Adam(model.parameters())

  # Load the weights, if they exist. If not, set up training files. 
  if force_restart or not weights_fpath.exists():
    print("[INFO] Train - Cold start; starting Tacotron training from scratch.")
    # Save the model once to put the first checkpoint there (checking
    # any writing issues before training)
    model.save(weights_fpath)

    # Embeddings metadata
    char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
    with open(char_embedding_fpath, "w", encoding="utf-8") as f:
      for symbol in symbols:
        if symbol == " ":
          symbol = "\\s" # For visual purposes.
        f.write("{}\n".format(symbol))
  else:
    print("[INFO] Train - Warm start; loading existing Tacotron weights at %s" % weights_fpath)
    model.load(weights_fpath, optimizer)
    print("[INFO] Train - Tacotron weights loaded successfully from step %d" % model.step)
  
  # Initialize the dataset. 
  metadata_fpath = syn_dir.joinpath("train.txt")
  mel_dir = syn_dir.joinpath("mels")
  embed_dir = syn_dir.joinpath("embeds")
  dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)

  # Main training loop. 
  print("\n[INFO] Train - Beginning training sessions...\n")

  # We train in 'sessions', with each session containing its own
  # learning hyperparameters. These are defined in hparams. 
  for i, session in enumerate(hparams.tts_schedule):
    current_step = model.get_step()
    r, lr, max_step, batch_size = session
    training_steps = max_step - current_step
    
    # Do we need to change to the next session? Or do we need to end?
    if current_step >= max_step:
      if i == len(hparams.tts_schedule) - 1:
        # We've completed training - save and exist. 
        print("\n[INFO] Train - Training complete. Saving model. Have a good night...")
        model.save(weights_fpath, optimizer)
        break
      else:
        # Head to the next session (next loop iteration)
        continue
    
    model.r = r

    # Begin the training for this session. 
    simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                  ("Batch Size", batch_size),
                  ("Learning Rate", lr),
                  ("Outputs/Step (r)", model.r)])
    
    for p in optimizer.param_groups:
      p["lr"] = lr
    
    # Create a dataloader for this session.
    collate_fn = partial(collate_synthesizer, r = r, hparams = hparams)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, 
                             num_workers=hparams.tts_dataloader_num_workers, 
                             collate_fn=collate_fn)
    
    # Use the profiler to provide training statistics.
    profiler = Profiler(summarize_every=10, disabled=False)

    total_iters = len(dataset)
    steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
    epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

    print("[INFO] Train - Beginning session. Loading data...")

    # Train on the dataset for this session. 
    for epoch in range(1, epochs+1):
      for i, (texts, mels, embeds, idx) in enumerate(data_loader, 1):
        start_time = time.time()
        profiler.tick("Blocking, waiting for batch (threaded)")

        # Generate stop tokens for training. 
        stop = torch.ones(mels.shape[0], mels.shape[2])
        for j, k in enumerate(idx):
          stop[j, :int(dataset.metadata[k][4])-1] = 0
        
        texts = texts.to(device)
        mels = mels.to(device)
        embeds = embeds.to(device)
        stop = stop.to(device)
        profiler.tick("Data sent to device: %s" % device)

        # Forward propagation. 

        # Includes parellization workaround due to python bug. 
        if device.type == "cuda" and torch.cuda.device_count() > 1:
          m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts, mels, embeds)
        else:
          m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)
        profiler.tick("Forward propagation complete")

        # Back-propagation. 
        
        # Get the loss and gradients. We get three loss values - m1, m2, 
        # and stop_loss. These all combine to get the total loss for this 
        # forward prop. 
        m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
        m2_loss = F.mse_loss(m2_hat, mels)
        stop_loss = F.binary_cross_entropy(stop_pred, stop)
        loss = m1_loss + m2_loss + stop_loss
        profiler.tick("Loss calculated")

        # Back-prop to get the gradients relative to J
        optimizer.zero_grad()
        loss.backward()
        profiler.tick("Back-propagation complete")

        # Gradient clipping. 
        if hparams.tts_clip_grad_norm is not None:
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
          if np.isnan(grad_norm.cpu()):
            print("[WARNING] Train - grad_norm was NaN!")
        
        # Apply the SGD update. 
        optimizer.step()
        profiler.tick("Parameter update complete")

        # Post-step actions

        # Bookkeeping updates.
        time_window.append(time.time() - start_time)
        loss_window.append(loss.item())

        # Print out statistics about this last step. 
        step = model.get_step()
        k = step
        msg = f" Step: {k} | Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | " \
              f"{1./time_window.average:#.2} steps/s  "
        stream(msg)

        # Backup or save the model as appropriate. 
        if backup_every != 0 and step % backup_every == 0:
          backup_fpath = weights_fpath.parent / f"synthesizer_{loss_window.average:#.4}_{k:06d}.pt"
          model.save(backup_fpath, optimizer)

        if save_every != 0 and step % save_every == 0 :
          # Must save latest optimizer state to ensure that resuming training
          # doesn't produce artifacts
          model.save(weights_fpath, optimizer)
        
        # Evaluation

        epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch # If the epoch is done
        step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0 # Every N steps

        # When we are ready to evaluate this particular step.
        if epoch_eval or step_eval:
          for sample_idx in range(hparams.tts_eval_num_samples):
            # Generate samples at most equal to the number in the batch. 
            if sample_idx + 1 <= len(texts):
              # Remove padding from mels using frame length in metadata. 
              mel_length = int(dataset.metadata[idx[sample_idx]][4])
              mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
              target_spectogram = np_now(mels[sample_idx]).T[:mel_length]
              attention_len = mel_length // model.r

              eval_model(attention = np_now(attention[sample_idx][:, :attention_len]),
                         mel_prediction = mel_prediction,
                         target_spectogram = target_spectogram,
                         input_seq = np_now(texts[sample_idx]),
                         step = step,
                         plot_dir = plot_dir,
                         mel_output_dir = mel_output_dir,
                         wav_dir = wav_dir,
                         sample_num = sample_idx+1,
                         loss = loss, 
                         hparams = hparams)
        
        profiler.tick("Extras (model saving, evaluation, etc) complete")

        # Break out of the loop to update training schedule if we're
        # done with this session. 
        if step >= max_step:
          break

      # After every epoch, add a line break. 
      print("")
  
# Evaluating a model during training. Note: this is sort of "pseudo
# evaluation" in that we're just comparing the output of the predicted
# spectogram to the solution of training data. 
def eval_model(attention, mel_prediction, target_spectogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
  # Save some results for evaluation.
  attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
  save_attention(attention, attention_path)

  # For debug purposes only, save the mel spectogram to disk. 
  mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
  np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

  # Also for debug, save the griffin lim inverted wav. To do this,
  # we need to first convert the mel spectogram to a wav.
  wav = audio.inv_mel_spectogram(mel_prediction.T, hparams)
  wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
  audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

  # Save the real and predicted mel spectogram plot to disk. 
  spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
  title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
  plot_spectogram(mel_prediction, str(spec_fpath), title=title_str, 
                  target_spectogram=target_spectogram, 
                  max_len=target_spectogram.size // hparams.num_mels)
  print("[DEBUG] Train - Input at step {}: {}".format(step, sequence_to_text(input_seq)))