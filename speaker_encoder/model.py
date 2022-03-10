#
# model.py
#
# Speaker Encoder model structure. A 3-layer LSTM network followed
# by a projection layer of 256 units (our output representation
# to be used)

from re import L
from speaker_encoder.model_params import *
from speaker_encoder.audio_params import *

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn

import numpy as np
import torch

# Class extends nn.Module. 
class SpeakerEncoder(nn.Module):

  # Expect the device + loss device to be passed in on init. Define
  # the network here. 
  def __init__(self, device, loss_device):

    # Init the nn.Module stuff.
    super().__init__()

    self.loss_device = loss_device

    # Define the network. Send each layer to the device after you
    # define it.
     
    # Our 3 LSTM layers of size 256.
    self.lstm = nn.LSTM(input_size=mel_n_channels,
                        hidden_size=model_hidden_size, 
                        num_layers=model_num_layers, 
                        batch_first=True).to(device)
                      
    # Our fully connected projection layer.
    self.linear = nn.Linear(in_features=model_hidden_size, 
                            out_features=model_embedding_size).to(device)
    
    # A final ReLU before normalization to make embeddings sparse, thus
    # more easily interpretable.
    self.relu = torch.nn.ReLU().to(device)

    # Cosine similarity scaling (with fixed initial parameter values).
    # This is the generalized end-to-end loss presented as part of GE2E -
    # specifically the affine transformation applied to the calculated
    # similarity matrix. 
    self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
    self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

    # Necessary if you're using the GPU (Not sure why it isn't for 
    # CPU usage. I'd blame that on pytorch). This explicitly requires
    # that calculated gradients are stored, so we can manipulate them.
    # They're non-leaf variables, so this is necessary. 
    self.similarity_weight.retain_grad()
    self.similarity_bias.retain_grad()

    # Loss. 
    self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

  # Gradient modifications. Reduce the scale of any gradient we
  # calculate and clip it to prevent gradient explosion.
  def do_gradient_ops(self):
    # Gradient scale
    self.similarity_weight.grad *= model_similarity_weight_grad_modifier
    self.similarity_bias.grad *= model_similarity_bias_grad_modifier

    # Gradient clipping
    clip_grad_norm_(self.parameters(), model_grad_clip, norm_type=2)

  # Computes the embeddings of a batch of utterance spectograms. 
  # For each batch, trickle through the network and return
  # ONLY the embedding, not the output of the output layer. 
  #
  # Takes in the batch of utterances as a tensor: 
  # (batch_size, n_frames, n_channels), as well the iniital hidden
  # state of the LSTM as a tensor:
  # (num_layers, batch_size, hidden_size) - defaults to tensor of
  # zeros if not applied.
  #
  # Returns the embeddings as a tensor:
  # (batch_size, embedding_size)
  def forward(self, utterances, hidden_init=None):
    # Pass the input through the LSTM layers and get all the outputs,
    # the final hidden state + cell state.
    out, (hidden, cell) = self.lstm(utterances, hidden_init)

    # We only grab the hidden state of the last layer. 
    embeds_raw = self.relu(self.linear(hidden[-1]))

    # L2 normalization of the embedding. Use 1e-5 as alpha.
    embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)

    return embeds

  # How we detemine wether our embedding is actually good or not.
  # Generalized End to End Loss (GE2E) that constructs a similarity
  # matrix indicating how similar certain spectograms are to each 
  # other. We use this matrix to derive our performance metric.
  #
  # Takes in the embeddings as a tensor: 
  # (speakers_per_batch, utterances_per_speaker, embedding_size)
  # 
  # Returns the similarity matrix itself as a tensor:
  # (speakers_per_batch, utterances_per_speaker, speakers_per_batch)
  def similarity_matrix(self, embeds):
    speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

    # Inclusive centroids (1 per speaker). Cloning is needed for reverse
    # differentiation
    centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
    centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

    # Exclusive centroids (1 per utterance)
    centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
    centroids_excl /= (utterances_per_speaker - 1)
    centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

    # Create the similarity matrix. The cosine similiarity of already
    # L2-normed vectors is simply the dot product of these vectors 
    # (just element-wise multiplication reduced by a sum). These
    # computations are vectorized for efficiency.

    # Create a zeroed tensor 
    sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                            speakers_per_batch).to(self.loss_device)
    mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
    for j in range(speakers_per_batch):
      mask = np.where(mask_matrix[j])[0]
      sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
      sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

    # Apply weights and biases. 
    sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
    return sim_matrix

  # Calculates the softmax loss (also according to GE2E). Uses the
  # similarity matrix to do so. 
  #
  # Takes in embeds as a tensor:
  # (speakers_per_batch, utterances_per_speaker, embedding_size)
  # 
  # Returns the loss + EER for this batch of embeddings, our surrogate
  # performance metric.
  def loss(self, embeds):
    # Get the similarity matrix first.
    sim_matrix = self.similarity_matrix(embeds)
    # Flatten the tensor into a proper matrix. 
    sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                    speakers_per_batch))
    ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
    target = torch.from_numpy(ground_truth).long().to(self.loss_device)

    # Calculate the loss. 
    loss = self.loss_fn(sim_matrix, target)

    # Calculate the Equal Error Rate (EER) - a common measurement 
    # used in biometric systems. 
    with torch.no_grad():
      # Grab the predictions versus labels in the correct format.
      inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
      labels = np.array([inv_argmax(i) for i in ground_truth])
      preds = sim_matrix.detach().cpu().numpy()

      # Snippet from https://yangcha.github.io/EER-ROC/
      fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
      eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return loss, eer