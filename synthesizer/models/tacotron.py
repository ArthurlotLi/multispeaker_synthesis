#
# tacotron.py
#
# Open-source implementation of Tacotron in pytorch, transcribed and 
# annotated by hand. This is professional grade work. 
#
# Include modifications described by the original paper to integrate
# the speaker encoder embedding into the encoder output frames. 
# Deviations from the model architecture described in the two
# reference papers are marked accordingly.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union

# Begin Encoder

# Two linear projection layers used within the decoder. One learns
# to output a scalar value, the other is effectively a fully connected
# layer with a ReLU + affine transformation. The scalar will be 
# flattened with sigmoid to range between 0 and 1. 
class HighwayNetwork(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.W1 = nn.Linear(size,size)
    self.W2 = nn.Linear(size, size)
    self.W1.bias.data.fill_(0.)
  
  # Forward propagation. Apply ReLU to the output. Weigh the output with
  # the scalar value fed into a sigmoid ranging between 0 and 1. 
  def forward(self, x):
    x1 = self.W1(x)
    x2 = self.W2(x)
    g = torch.sigmoid(x2)
    y = g * F.relu(x1) + (1. - g) * x
    return y

# Encoder. Given input text as a sequence, executes character embedding
# and passes the result through CNN layers, GRU* Layers, and 
# concatenates the result with the speaker embedding. This version
# of tacotron also includes a prenet. 
class Encoder(nn.Module):
  def __init__(self, embed_dims, num_chars, encoder_dims, K, num_highways, dropout):
    super().__init__()
    prenet_dims = (encoder_dims, encoder_dims)
    cbhg_channels = encoder_dims
    self.embedding = nn.Embedding(num_chars, embed_dims)
    self.pre_net = PreNet(embed_dims, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1],
                          dropout=dropout)
    self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                     proj_channels=[cbhg_channels, cbhg_channels],
                     num_highways=num_highways)
    
  # Forward propagation. Given input x, embed characters as vectors. 
  # With the embeddings, go through the preNet, CBHG to get the
  # encoded frames. Append the speaker embedding to all output
  # frames. 
  def forward(self, x, speaker_embedding=None):
    x = self.embedding(x)
    x = self.pre_net(x)
    x.transpose_(1,2)
    x = self.cbhg(x)
    if speaker_embedding is not None:
      x = self.add_speaker_embedding(x, speaker_embedding)
    return x
  
  # Concat a speaker embedding to all output frames of the encoder. 
  # x: encoder output - 3D Tensor of size:
  #   (batch_size, num_chars, tts_embed_dims)
  # speaker_embedding - 2D (train) or 1D (inference) tensor of sizes:
  #   (batch_size, speaker_embedding_size) / (speaker_embedding_size)
  def add_speaker_embedding(self, x, speaker_embedding):
    # Save the dimensions so they make sense. 
    batch_size = x.size()[0]
    num_chars = x.size()[1]

    # Depending on whether we're in training or inference.
    if speaker_embedding.dim() == 1:
      idx=0
    else:
      idx=1

    # Make a copy of each speaker embedding to match the input text
    # length. Results in a tensor: 
    #   (batch_size, num_chars * tts_embed_dimsidx)
    speaker_embedding_size = speaker_embedding.size()[idx]
    e = speaker_embedding.repeat_interleave(num_chars, dim=idx)

    # Reshape + transpose for concatenation. Results in a tiled
    # speaker embedding.
    e = e.reshape(batch_size, speaker_embedding_size, num_chars)
    e = e.transpose(1, 2)

    # Concatenate the speaker embedding with the encoder output.
    x = torch.cat((x, e), 2)
    return x

# 1D convolutional layer with batch normalization and ReLU. Takes in
# an embedding and outputs an representation of that embedding.
class BatchNormConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, relu=True):
    super().__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
    self.bnorm = nn.BatchNorm1d(out_channels)
    self.relu = relu
  
  # Conv1d -> ReLU -> batch norm
  def forward(self, x):
    x = self.conv(x)
    x = F.relu(x) if self.relu is True else x
    return self.bnorm(x)

# CBHG - Convolutional + Recurrent layers for the encoder. 
class CBHG(nn.Module):
  def __init__(self, K, in_channels, channels, proj_channels, num_highways):
    super().__init__()

    # List of all recurrent neural networks to call "flatten_parameters"
    self._to_flatten = []

    # Indices of all kernels. 
    self.bank_kernels = [i for i in range(1, K+1)]
    # Where we'll keep all of our CNN layers.
    self.conv1d_bank = nn.ModuleList()
    for k in self.bank_kernels:
      # Fill the bank of CNN layers. 
      conv = BatchNormConv(in_channels, channels, k)
      self.conv1d_bank.append(conv)
    
    # Define a max pooling layer
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

    # Create two CNN layers as projection layers.
    self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
    self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

    # Fix highway input if necessary.
    if proj_channels[-1] != channels:
      self.highway_mismatch = True
      self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
    else:
      self.highway_mismatch = False
    
    # Define a "Highway Network"
    self.highways = nn.ModuleList()
    for i in range(num_highways):
      hn = HighwayNetwork(channels)
      self.highways.append(hn)
    
    # Define a bidirectional GRU layer to follow the Convolutional
    # layers. (Was a LSTM layer in the papers) This will need to be
    # flattened.
    self.rnn = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)
    self._to_flatten.append(self.rnn)

    # Avoid fragmentation of RNN parameters and associated warning.
    self._flatten_parameters()
  
  # Forward propagation for this component. Go through the CNN layers,
  # max pool, project the results twice, pass through the highways,
  # then finally pass through the bidirectional GRU to output the
  # encoded output frames. 
  def forward(self, x):
    # We did call _flatten_parameters on init, but when using
    # DataParallel the model gets replicated - thus it is no longer
    # guaranteed that the weights are contiguous in GPU memory. So,
    # we need to call it again here.
    self._flatten_parameters()

    # Save these for later. 
    residual = x
    seq_len = x.size(-1)
    conv_bank = []

    # Convolution Bank
    for conv in self.conv1d_bank:
      c = conv(x) # Apply convolution.
      conv_bank.append(c[:, :, :seq_len])
    
    # Stack along the channel axis
    conv_bank = torch.cat(conv_bank, dim=1)

    # Dump the last padding to fit residual vector.
    x = self.maxpool(conv_bank)[:, :, :seq_len]

    # Conv1d projections - twice. 
    x = self.conv_project1(x)
    x = self.conv_project2(x)

    # Residual Connect
    x = x + residual

    # Pass through the highways
    x = x.transpose(1,2)
    if self.highway_mismatch is True:
      x = self.pre_highway(x)
    for h in self.highways: x = h(x)

    # Finally, pass through the bidirectional GRU.
    x, _ = self.rnn(x)

    return x
  
  # Calls 'flatten_parameters' on all RNNs. Used to improve efficiency
  # on the GPU itself as well as to avoid PyTorch yelling at us. 
  def _flatten_parameters(self):
    [m.flatten_parameters() for m in self._to_flatten]

# A 2 layer prenet applied to the encoder and decoder. Includes
# dropout 0.5.
class PreNet(nn.Module):
  def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
    super().__init__()
    self.fc1 = nn.Linear(in_dims, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.p = dropout
  
  # Forward propagation. Pretty straightforward.
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = F.dropout(x, self.p, training=True)
    x = self.fc2(x)
    x = F.relu(x)
    x = F.dropout(x, self.p, training=True)
    return x

# End Encoder 

# Begin Attention

# Our attention component. This is a location-sensitive bridge between
# the encoder and the decoder. It consists of two linear layers that
# do some kinda whack stuff during forward propagation.
class Attention(nn.Module):
  def __init__(self, attn_dims):
    super().__init__()
    self.W = nn.Linear(attn_dims, attn_dims, bias=False)
    self.v = nn.Linear(attn_dims, 1, bias=False)
  
  def forward(self, encoder_seq_proj, query, t):
    # Transform the query vector
    query_proj = self.W(query).unsqueeze(1)

    # Compute the scores
    u = self.v(torch.tanh(encoder_seq_proj + query_proj))
    scores = F.softmax(u, dim=1)

    return scores.transpose(1, 2)

# What we call the LSA component - our attention network that
# bridges between the decoder and the attention component. 
class LSA(nn.Module):
  def __init__(self, attn_dim, kernel_size = 31, filters = 32):
    super().__init__()
    self.conv = nn.Conv1d(1, filters, padding=(kernel_size - 1) // 2, kernel_size = kernel_size, bias=True)
    self.L = nn.Linear(filters, attn_dim, bias=False)
    self.W = nn.Linear(attn_dim, attn_dim, bias=True) # Include attention bias in this term.
    self.v = nn.Linear(attn_dim, 1, bias=False)
    self.cumulative = None
    self.attention = None
  
  # Initialize, given the projection of the encoder. This is the
  # first step of the attention feedback loop.
  def init_attention(self, encoder_seq_proj):
    # Use the same device as the parameters of this component.
    device = next(self.parameters()).device 
    b, t, c = encoder_seq_proj.size()
    self.cumulative = torch.zeros(b, t, device=device)
    self.attention = torch.zeros(b, t, device=device)
  
  def forward(self, encoder_seq_proj, query, t, chars):
    if t == 0: self.init_attention(encoder_seq_proj)

    processed_query = self.W(query).unsqueeze(1)

    location = self.cumulative.unsqueeze(1)
    processed_loc = self.L(self.conv(location).transpose(1, 2))

    u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
    u = u.squeeze(-1)

    # Mask zero padding chars
    u = u * (chars != 0).float()

    # Smooth attention
    scores = F.softmax(u, dim=1)
    self.attention = scores
    self.cumulative = self.cumulative + self.attention

    return scores.unsqueeze(-1).transpose(1, 2)

# End Attention

# Begin Decoder

# Interfaces with the Attention component using the LSA component. 
# Takes in input frames concatenated with the prenet-processed
# previous generated frame. Calculates attention scores and 
# ultimately, through two RNN layers, creates two projections -
# one generating the output mel spectogram, the other outputting
# a stop token. 
class Decoder(nn.Module):
  max_r = 20
  def __init__(self, n_mels, encoder_dims, decoder_dims, lstm_dims, 
               dropout, speaker_embedding_size):
    super().__init__()
    self.register_buffer("r", torch.tensor(1, dtype=torch.int))
    self.n_mels = n_mels
    prenet_dims = (decoder_dims * 2, decoder_dims * 2)
    # Our prenet that feeds previous output back in autoregressively.
    self.prenet = PreNet(n_mels, fc1_dims=prenet_dims[0], fc2_dims=prenet_dims[1],
                         dropout=dropout)
    # Our attention network. 
    self.attn_net = LSA(decoder_dims)
    self.attn_rnn = nn.GRUCell(encoder_dims + prenet_dims[1] + speaker_embedding_size, decoder_dims)
    self.rnn_input = nn.Linear(encoder_dims + decoder_dims + speaker_embedding_size, lstm_dims)
    # Two LSTM layers
    self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
    self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
    # Our two projections - one for the mel spectogram, the other for
    # the scalar stop token. 
    self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
    self.stop_proj = nn.Linear(encoder_dims + speaker_embedding_size + lstm_dims, 1)
  
  def zoneout(self, prev, current, p=0.1):
    # Use the same device as the parameters of this component
    device = next(self.parameters()).device 
    mask = torch.zeros(prev.size(), device=device).bernoulli_(p)
    return prev * mask + current * (1 - mask)
  
  # Forward propagation. PreNet -> attention RNN -> attention
  # scores -> two RNN layers -> project Mel + stop token.
  def forward(self, encoder_seq, encoder_seq_proj, prenet_in, 
              hidden_states, cell_states, context_vec, t, chars):
    # Need this for reshaping mel spectograms.
    batch_size = encoder_seq.size(0)

    # Unpack hidden + cell states. 
    attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
    rnn1_cell, rnn2_cell = cell_states

    # PreNet for the attention RNN.
    prenet_out = self.prenet(prenet_in)

    # Compute the Attention RNN hidden state. This takes in the
    # ouptut of the PreNet (processed previous output) concatenated
    # with the input vector.
    attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
    attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

    # Compute the attention scores.
    scores = self.attn_net(encoder_seq_proj, attn_hidden, t, chars)

    # Dot product the scores to create a complete context vector. Fun
    # fact, "@" may represent the dot product.
    context_vec = scores @ encoder_seq
    context_vec = context_vec.squeeze(1)

    # Concat attention RNN output with context vector and projection.
    x = torch.cat([context_vec, attn_hidden], dim=1)
    x = self.rnn_input(x)

    # Compute the first Residual RNN. If we're training, apply a mask.
    rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
    if self.training:
      rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
    else:
      rnn1_hidden = rnn1_hidden_next
    x = x + rnn1_hidden

    # Compute the second Residual RNN. Same deal with zoneout.
    rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
    if self.training:
      rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
    else:
      rnn2_hidden = rnn2_hidden_next
    x = x + rnn2_hidden
  
    # Project the Mel Spectogram frame. 
    mels = self.mel_proj(x)
    mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
    hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
    cell_states = (rnn1_cell, rnn2_cell)

    # Project the stop token. 
    s = torch.cat((x, context_vec), dim=1)
    s = self.stop_proj(s)
    stop_tokens = torch.sigmoid(s)

    return mels, scores, hidden_states, cell_states, context_vec, stop_tokens

# End Decoder

# Begin Full Model

# Put everything together with the complete Tacotron model. This is
# hands down the most sophisticated freaking model I've ever seen
# in my life. 
#
# Full model: 
# Encoder -> Encoder Projection -> Decoder -> PostNet -> Post Projection
class Tacotron(nn.Module):
  def __init__(self, embed_dims, num_chars, encoder_dims, decoder_dims, n_mels,
               fft_bins, postnet_dims, encoder_K, lstm_dims, postnet_K, num_highways,
               dropout, stop_threshold, speaker_embedding_size):
    super().__init__()
    self.n_mels = n_mels
    self.lstm_dims = lstm_dims
    self.encoder_dims = encoder_dims
    self.decoder_dims = decoder_dims
    self.speaker_embedding_size = speaker_embedding_size
    
    self.encoder = Encoder(embed_dims, num_chars, encoder_dims, 
                           encoder_K, num_highways, dropout)
    self.encoder_proj = nn.Linear(encoder_dims + speaker_embedding_size, decoder_dims, bias=False)
    self.decoder = Decoder(n_mels, encoder_dims, decoder_dims, lstm_dims,
                           dropout, speaker_embedding_size)
    self.postnet = CBHG(postnet_K, n_mels, postnet_dims, 
                        [postnet_dims, fft_bins], num_highways)
    self.post_proj = nn.Linear(postnet_dims, fft_bins, bias=False)

    self.init_model()
    self.num_params()

    self.register_buffer("step", torch.zeros(1, dtype=torch.long))
    self.register_buffer("stop_threshold", torch.tensor(stop_threshold, dtype=torch.float32))
  
  @property
  def r(self):
    return self.decoder.r.item()
  
  @r.setter
  def r(self, value):
    self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)
  
  # Full forward propagation of the model. Initialize states first.
  # Run the encoder with speaker embedding and project. Run the
  # decoder loop with attention. Concat the output mel spectogram
  # frames into one sequence, post-process them with the postNet.
  def forward(self, x, m, speaker_embedding):
    # Ensure you're using the same device as the parameters.
    device = next(self.parameters()).device

    self.step += 1
    batch_size, _, steps = m.size()

    # Initialize all hidden states and pack into a tuple. 
    attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
    rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
    rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
    hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

    # Initialize all LSTM cell states and pack into a tuple.
    rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
    rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
    cell_states = (rnn1_cell, rnn2_cell)

    # "GO" Frame for start of decoder loop, as we won't have anything
    # to provide the PreNet otherwise. 
    go_frame = torch.zeros(batch_size, self.n_mels, device=device)

    # Need an initial context vector. 
    context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

    # Be sure to run here with the speaker embedding! Otherwise this is
    # all kinda pointless. 
    encoder_seq = self.encoder(x, speaker_embedding)

    # Projecting the results gets rid of unecessary matrix 
    # multiplication during decoding.
    encoder_seq_proj = self.encoder_proj(encoder_seq)

    # Run the decoder loop
    mel_outputs, attn_scores, stop_outputs = [], [], []
    for t in range(0, steps, self.r):
      # Provide the last input to the PreNet, or the go_frame if 
      # this is the first run.
      prenet_in = m[:, :, t-1] if t > 0 else go_frame
      mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
        self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states,
                     cell_states, context_vec, t, x)
      mel_outputs.append(mel_frames)
      attn_scores.append(scores)
      stop_outputs.extend([stop_tokens] * self.r)
  
    # Given the outputs, combine all into a mel spectogram sequence.
    mel_outputs = torch.cat(mel_outputs, dim=2)

    # Post-process the spectograms by passing through the PostNet +
    # post projection. 
    postnet_out = self.postnet(mel_outputs)
    linear = self.post_proj(postnet_out)
    linear = linear.transpose(1,2)

    # A few things for easy visualization:
    attn_scores = torch.cat(attn_scores, 1)
    stop_outputs = torch.cat(stop_outputs, 1)

    # And we're done with one forward prop! Whew...
    return mel_outputs, linear, attn_scores, stop_outputs

  # For inference! A few important distinctions from forward. The 
  # principal difference is that we're only going to be calculating
  # results for a single sequence of text, not a minibatch. 
  def generate(self, x, speaker_embedding=None, steps=2000):
    # Enter evaluation mode. 
    self.eval()
    device = next(self.parameters()).device

    batch_size, _ = x.size()

    # Need to initialize all hidden states and pack into a tuple. 
    attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
    rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
    rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
    hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

    # Need to initialise all lstm cell states and pack into tuple.
    rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
    rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
    cell_states = (rnn1_cell, rnn2_cell)

    # An inital "GO" Frame for the decoder loop. 
    go_frame = torch.zeros(batch_size, self.n_mels, device=device)

    # Initial context vector.
    context_vec = torch.zeros(batch_size, self.encoder_dims + self.speaker_embedding_size, device=device)

    # Include the embedding, and then project it to avoid unecessary
    # matrix multiplication. 
    encoder_seq = self.encoder(x, speaker_embedding)
    encoder_seq_proj = self.encoder_proj(encoder_seq)

    # Run the decoder loop.
    mel_outputs, attn_scores, stop_outputs = [], [], []
    for t in range(0, steps, self.r):
      prenet_in = mel_outputs[-1][:, :, -1] if t > 0 else go_frame
      mel_frames, scores, hidden_states, cell_states, context_vec, stop_tokens = \
        self.decoder(encoder_seq, encoder_seq_proj, prenet_in, hidden_states,
                     cell_states, context_vec, t, x)
      mel_outputs.append(mel_frames)
      attn_scores.append(scores)
      stop_outputs.extend([stop_tokens] * self.r)

      # Actually stop the loop when all stop tokens in batch exceed the
      # threshold of 0.5. 
      if (stop_tokens > 0.5).all() and t > 10: break
    
    # Concat the mel outputs into a sequence.
    mel_outputs = torch.cat(mel_outputs, dim=2)

    # Post processing.
    postnet_out = self.postnet(mel_outputs)
    linear = self.post_proj(postnet_out)
    linear = linear.transpose(1, 2)

    # A few things for easy visualization:
    attn_scores = torch.cat(attn_scores, 1)
    stop_outputs = torch.cat(stop_outputs, 1)

    # And we're done! Whew. 
    return mel_outputs, linear, attn_scores

  def init_model(self):
    for p in self.parameters():
      if p.dim() > 1: nn.init.xavier_uniform_(p)

  def get_step(self):
    return self.step.data.item()

  # Assignment to parameters or buffers if overloaded; updates internal
  # dict entry.
  def reset_step(self):
    self.step = self.step.data.new_tensor(1)
  
  # Logs status to file. 
  def log(self, path, msg):
    with open(path, "a") as f:
      print(msg, file=f)
  
  # Loads the model from a checkpoint. If we're training, provide the
  # optimizer object and the optimizer state will be restored as well.
  def load(self, path, optimizer=None):
    device = next(self.parameters()).device
    checkpoint = torch.load(str(path), map_location = device)
    self.load_state_dict(checkpoint["model_state"])

    if "optimizer_state" in checkpoint and optimizer is not None:
      optimizer.load_state_dict(checkpoint["optimizer_state"])

  # Saves the model as you'd expect. Make sure you save the optimizer
  # state as well by providing it to this method. 
  def save(self, path, optimizer = None):
    if optimizer is not None:
      torch.save({
        "model_state" : self.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
      }, str(path))
    else:
      torch.save({
        "model_state" : self.state_dict()
      }, str(path))

  # Gets the total number of parameters present in the model. 
  # Hint... it's a lot. 
  def num_params(self, print_out = True):
    parameters = filter(lambda p: p.requires_grad, self.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
      print("[INFO] Tacotron - Trainable parameters: %.3fM" % parameters)
    return parameters

# End Full Model
