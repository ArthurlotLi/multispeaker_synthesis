#
# plot.py
#
# Utilities for plotting mel spectograms + alignment.

import numpy as np

# Splits any string based on a specific character, returning it
# with the string + maximum number of words allowed.
def _split_title_line(title_text, max_words = 5):
  seq = title_text.split()
  return "\n".join([" ".join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])

# Plot alignment during training. Expects the alignment b/w encoder + 
# decoder as well as the path to save the output graph.
def plot_alignment(alignment, path, title=None, split_title=False, max_len = None):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  if max_len is not None:
    alignment = alignment[:, :max_len]
  
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_suplot(111)

  im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
  fig.colorbar(im, ax=ax)
  xlabel = "Decoder Timestep"
  ylabel = "Encoder Timestep"

  if split_title:
    title = _split_title_line(title)
  
  plt.xlabel(xlabel)
  plt.title(title)
  plt.ylabel(ylabel)
  plt.tight_layout()
  plt.savefig(path, format="png")
  plt.close()

# Plots a mel spectogram. Expects the spectogram + output location.
# Optionally may also print the target spectogram alongside the 
# predicted for visual comparison of how well the model is doing.
def plot_spectogram(pred_spectogram, path, title=None, split_title=False, 
                    target_spectogram=None, max_len=None, auto_aspect=False):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  # Hah! Found a bug. 
  if max_len is not None:
    pred_spectogram = pred_spectogram[:max_len]
    if target_spectogram is not None:
      target_spectogram = target_spectogram[:max_len]
  
  # And another bug.
  if title is None:
    title = "Mel-Spectogram"

  if split_title:
    title = _split_title_line(title)
  
  fig = plt.figure(figsize=(10,8))
  # Set common labels
  fig.text(0.5, 0.18, title, horizontalalignment="center", fontsize=16)
  
  # Target spectogram subplot
  if target_spectogram is not None:
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)

    if auto_aspect:
      im = ax1.imshow(np.rot90(target_spectogram), aspect="auto", interpolation="none")
    else:
      im = ax1.imshow(np.rot90(target_spectogram), interpolation="none")
    
    ax1.set_title("Target Mel-Spectogram")
    fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
    ax2.set_title("Predicted Mel-Spectogram")
  else:
    ax2 = fig.add_subplot(211)
  
  if auto_aspect:
    im = ax2.imshow(np.rot90(pred_spectogram), aspect = "auto", interpolation = "none")
  else:
    im = ax2.imshow(np.rot90(pred_spectogram), interpolation="none")
  fig.colorbar(mappable=im, shrink= 0.65, orientation="horizontal", ax=ax2)

  plt.tight_layout()
  plt.savefig(path, format="png")
  plt.close()