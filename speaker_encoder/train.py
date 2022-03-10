#
# train.py
#
# Speaker Encoder training harness. Executes the trianing of the model
# As you'd expect. 

from speaker_encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from speaker_encoder.model import SpeakerEncoder
from speaker_encoder.model_params import *
from speaker_encoder.visualizations import Visualizations
from utils.profiler import Profiler

from pathlib import Path
import os
import torch

# As cuda operations are async, synchronize for correct profiling.
def sync(device: torch.device): 
  if device.type == "cuda":
    torch.cuda.synchronize(device)

# Execute training. 
def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int, 
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
  # Create the dataset + dataloader objects. 
  dataset = SpeakerVerificationDataset(clean_data_root)
  loader = SpeakerVerificationDataLoader(
    dataset,
    speakers_per_batch,
    utterances_per_speaker,
    num_workers = data_loader_num_workers,
  )

  # Setup the device on which to run forward pass + loss calculations.
  # Note that these CAN be different devices, as the forward pass is
  # faster on the GPU whereas the loss (depending on what
  # hyperparameters you chose) is often faster on the CPU.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #loss_device = torch.device("cpu")
  loss_device = device

  # Create the model + optimizer. We'll be using Adam.
  model = SpeakerEncoder(device, loss_device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
  init_step = 1

  # Set the file path for the model given input arguments. Create it if
  # it does not exist. 
  model_dir = models_dir / run_id
  model_dir.mkdir(exist_ok=True, parents=True)
  state_fpath = model_dir / "encoder.pt"

  # Set model into training mode. 
  model.train()

  # Load an existing model if we find one that matches. We'll train it
  # as a warm start if we do. 
  if not force_restart:
    if state_fpath.exists():
      # Clean the path first, in case of a mismatch between / and \ to
      # avoid pytorch load errors.
      state_fpath = os.path.realpath(state_fpath)

      print("[INFO] Train - Found existing model \"%s\" at %s, loading it and resuming training (warm start)." % (run_id,state_fpath))

      # Found it! Load the checkpoint and set the first step #.
      checkpoint = torch.load(state_fpath)
      init_step = checkpoint["step"]

      # Load from the checkpoint accordingly. Note: Legacy checkpoints
      # may have different names.
      if "model_state" not in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
      else:
        model.load_state_dict(checkpoint["model_state"])
      if "optimizer_state" not in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      else:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

      # Reset the learning rate to init since this is a warm start.
      optimizer.param_groups[0]["lr"] = learning_rate_init
    else:
      print("[INFO] Train - No existing model \"%s\" was found. Beginning training from scratch." % run_id)
  else:
    print("[INFO] Train - Force_restart set to true. Beginning training from scratch.")

  # Set model into training mode. 
  model.train()

  # Initialize the visualization environment
  vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
  vis.log_dataset(dataset)
  vis.log_params()

  # Provide the device name to visdom.
  device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
  vis.log_implementation({"Device": device_name})

  # Training Loop.

  # Use the profiler to provide training statistics.
  profiler = Profiler(summarize_every=10, disabled=False)

  # Go through epochs.
  for epoch in range(1, model_epochs):
    print("[INFO] Training - Epoch %d/%d" % (epoch, model_epochs+1) )

    # Go through the training dataset each epoch. There's a LOT
    # of data here. 
    for step, speaker_batch in enumerate(loader, init_step):
      profiler.tick("Blocking, waiting for batch (threaded)")

      # Forward propagation. 

      # Send the next batch to the device and propagate them through 
      # the network.
      inputs = torch.from_numpy(speaker_batch.data).to(device)
      sync(device)
      profiler.tick("Data sent to device: %s" % device)

      # Get the embedding vector output by the model. 
      embeds = model(inputs)
      sync(device)
      profiler.tick("Forward propagation complete")

      # Calculate the loss with the loss device (Our CPU, not GPU) and
      # apply it. 
      embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
      loss, eer = model.loss(embeds_loss)
      sync(loss_device)
      profiler.tick("Loss calculated")

      # Back-propagation.

      # Calculate the gradient relative to the calculated loss. 
      model.zero_grad()
      loss.backward()
      profiler.tick("Back-propagation complete")

      # Apply the update appropriately given the gradient. 
      # "do_graident_ops", defined in model.py, applies gradient scaling
      # and gradient clipping before the step is applied. 
      model.do_gradient_ops()
      optimizer.step()
      profiler.tick("Parameter update complete")

      # Visualization.

      # Update visualizations given the performance + update of the model.
      vis.update(loss.item(), eer, step)

      # Draw projections and save them to the backup folder. Do this
      # only ever <umap_every> few steps.
      if umap_every != 0 and step % umap_every == 0:
        print("[DEBUG] Train - Drawing and saving projections (step %d)" % step)
        # Filename is umap_<step>.png.
        projection_fpath = model_dir / f"umap_{step:06d}.png"
        # Visualize the embeddings.
        embeds = embeds.detach().cpu().numpy()
        vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
        vis.save()

      # Model Saving.
      
      # Save the model, overwriting the latest version. Do this only
      # every <save_every> few steps so you don't slow training down
      # too much.
      if save_every != 0 and step % save_every == 0:
        print("[DEBUG] Train - Saving model (step %d)" % step)
        torch.save({
          "step": step + 1,
          "model_state": model.state_dict(),
          "optimizer_state": optimizer.state_dict(),
        }, state_fpath)
      
      # Save a backup - essentially a model checkpoint independent
      # from model saving. 
      if backup_every != 0 and step % backup_every == 0:
        print("[DEBUG] Train - Performing model backup (step %d)" % step)
        backup_fpath = model_dir / f"encoder_{eer:06f}_{loss:06f}_{step:07d}.pt" # .bak means it won't overwrite. 
        torch.save({
          "step": step + 1,
          "model_state": model.state_dict(),
          "optimizer_state": optimizer.state_dict(),
        }, backup_fpath)

      # Ready to go again. 
      profiler.tick("Visualization, model saving/checkpointing complete")
