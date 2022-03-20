#
# test.py
#
# Enables chain testing of a folder full of synthesizer checkpoints 
# in order to obtain a complete picture of train + test loss, as
# well as to decisively return the best performing checkpoint in 
# lieu of early stopping. Might be a bit more painful than the
# speaker_encoder or trigger word detection as these models are 
# >300mb.
#
# Expects the testing data to be have been preprocessed already,
# with both the audio and the embeds ready. Also expects the testing
# data to be relatively small, with the intent to load every model
# and run forward pass for every model only once. 

import synthesizer.audio_utils as audio
from synthesizer.tacotron import Tacotron
from synthesizer.hparams import hparams
from synthesizer.utils.symbols import symbols
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer

from pathlib import Path
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

_model_suffix = ".pt"
_file_name_prefix = "chain_test_results_"
_file_name_suffix = ".txt"

# This reduction factor should match those in the tts_schedule
# hyperparameter. 
_model_r = 2

_batch_size = 32
_num_dataloader_workers = 1 # Data loading is SUPER fast. 

# Principal function for executing batch model testing, given a
# path to a folder containing models. Will execute testing with all
# files in this folder with the .pt suffix. Also requires the
# report directory.
#
# Expects each model file to be labeled as follows:
#   synthesizer_<loss>_<step>.pt
#   Ex) synthesizer_0.2816_044400.pt
def batch_test(model_batch_dir: Path, clean_data_root: Path, 
               test_report_dir: Path, use_cpu: bool):
  print("[INFO] Test - Initializing batch test.")

  filename_uid = 0
  chain_test_results = {}
  chain_test_results_acc_map = {}

  results_folder_contents = os.listdir(test_report_dir)
  result_index = 0
  for file in results_folder_contents:
    if file.endswith(".txt"):
      file_number_str = file.replace(_file_name_prefix, "").replace(_file_name_suffix, "")
      file_number = -1
      try:
        file_number = int(file_number_str)
        if(file_number >= result_index):
          result_index = file_number + 1
      except:
        pass # Ignore any files that aren't of the correct format. 
  
  print("[DEBUG] Test - Processing result index %d." % result_index)

  # Gathers all of the models in the directory. 
  files = os.listdir(model_batch_dir)
  files_models = []

  # Throw all applicable files into a list.
  for i in range(0, len(files)):
    filename = files[i]

    if filename.endswith(_model_suffix):
      files_models.append(filename)

  print("[INFO] Test - Found %d models in directory." % len(files_models))

  # Initialize the dataset. 
  metadata_fpath = clean_data_root.joinpath("train.txt")
  mel_dir = clean_data_root.joinpath("mels")
  embed_dir = clean_data_root.joinpath("embeds")
  dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)

  # Setup the device on which to run forward pass + loss calculations.
  device = None
  if use_cpu:
    device = torch.device("cpu")
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Work through each model. 
  for i, filename in tqdm(enumerate(files_models), desc="Models Tested", total=len(files_models)):
    loss = None

    try:
      # Use batch + filename. 
      loss = _test_model_worker(model_batch_dir + "/" + file, dataset, device)
    except Exception as e:
      print("\n\n[WARNING] Test - Encountered exception while testing file: %s" % file)
      print(e, end="\n\n")

    if loss is None:
      print("[WARNING] Test - Received empty loss!")
      chain_test_results[filename_uid] = "00.00000000 - " + str(filename) + " TEST FAILED!\n"
      chain_test_results_acc_map[-1] = filename_uid
    else:
      chain_test_results[filename_uid] = "%.8f - " % (loss*100) + str(filename) + "\n"
      # If a model of that exact accuracy exists already, append
      # a tiny number to it until it's unique. 
      if loss in chain_test_results_acc_map:
        sorting_acc = None
        while sorting_acc is None:
          loss = loss + 0.000000000000001 # Acc has 15 decimal precision. Append by 1 to break ties.
          if loss not in chain_test_results_acc_map:
            sorting_acc = loss
        chain_test_results_acc_map[sorting_acc] = filename_uid
      else:
        chain_test_results_acc_map[loss] = filename_uid
      
      filename_uid = filename_uid + 1
  
  if(filename_uid == 0):
    print("[WARNING] Test - No files with suffix %s found at location \"%s\". Please specify another location with an argument or move/copy the model(s) accordingly." % (_model_suffix,model_batch_dir))
    return

  # Sort the results. 
  chain_test_results_acc_map = dict(sorted(chain_test_results_acc_map.items(), key=lambda item: item[0]))

  # With the results, write to file. 
  _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map, result_index) 


# Evaluate a model, given the location of the model, and the batch.
def _test_model_worker(model_location, dataset, device):
  # Load the model with the device.
  model = Tacotron(embed_dims=hparams.tts_embed_dims,
                   num_chars=len(symbols),
                   encoder_dims=hparams.tts_encoder_dims,
                   decoder_dims=hparams.tts_decoder_dims,
                   n_mels=hparams.num_mels,
                   fft_bins=hparams.num_mels,
                   postnet_dims=hparams.tts_postnet_dims,
                   encoder_K=hparams.tts_encoder_K,
                   lstm_dims=hparams.tts_lstm_dims,
                   postnet_K=hparams.tts_postnet_K,
                   num_highways=hparams.tts_num_highways,
                   dropout=hparams.tts_dropout,
                   stop_threshold=hparams.tts_stop_threshold,
                   speaker_embedding_size=hparams.speaker_embedding_size).to(device)
    
  model.load(model_location)

  # Always set the model into evaluation/inference mode.
  model.eval()

  print("[INFO] Test - Loaded synthesizer \"%s\" trained to step %d" % (model_location, model.state_dict()["step"]))

  # Create a dataloader for this session.
  collate_fn = partial(collate_synthesizer, r = _model_r, hparams = hparams)
  loader = DataLoader(dataset, 
                      _batch_size,
                      _num_dataloader_workers, 
                      collate_fn=collate_fn)

  total_loss = 0

  for step, (texts, mels, embeds, idx) in enumerate(loader):
    # Generate stop tokens for training. 
    stop = torch.ones(mels.shape[0], mels.shape[2])
    for j, k in enumerate(idx):
      stop[j, :int(dataset.metadata[k][4])-1] = 0
    
    texts = texts.to(device)
    mels = mels.to(device)
    embeds = embeds.to(device)
    stop = stop.to(device)

    # Forward propagation. 
    m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

    # Calculate the loss.
    m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
    m2_loss = F.mse_loss(m2_hat, mels)
    stop_loss = F.binary_cross_entropy(stop_pred, stop)
    loss = m1_loss + m2_loss + stop_loss

    # Append to our total loss. 
    total_loss += loss

  # Return the result.
  return total_loss

# Writes results of batch testing to file. 
def _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map, result_index):
  try:
    filename = test_report_dir.joinpath(_file_name_prefix + str(result_index) + _file_name_suffix)
    print("\n[INFO] Test - Chain test complete. Writing results to file '%s'..." % filename)
    f = open(filename, "w")

    f.write("=================================\nMultispeaker Synthesis - Synthesizer Test Results\n=================================\n\n")
    # Write results of each model, sorted.
    for key in chain_test_results_acc_map:
      f.write(chain_test_results[chain_test_results_acc_map[key]])

    f.close()
  except Exception as e:
    print("[ERROR] Failed to write results to file! Exception:")
    print(e)
  
  _graph_train_test_history(chain_test_results, chain_test_results_acc_map, filename)

  print("[INFO] Test - Write complete. Have a good night...")

# Given all the test loss that we've generated, let's 
# graph every single one. Note that we expect a very specific
# file name structure for these model iterations:
#
# Expects full location of the file that has been written -
# the suffix will be stripped but otherwise the graph will be
# written to file with that exact name. 
def _graph_train_test_history(chain_test_results, chain_test_results_acc_map, filename):
  print("[INFO] Test - Generating test loss history graph.")

  graph_width_inches = 13
  graph_height_inches = 7
  
  graph_location = str(filename).replace(".txt", "")

  title = "Multispeaker Synthesis - Synthesizer - Train + Test History"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)

  # Gather points. Every point should be indexed by step. 
  indices = []
  test_losses = []
  train_losses = []

  # Each item should be structured as such:
  # 99.69913960 - synthesizer_0.472423_0211800.pt
  for key in chain_test_results_acc_map: 
    string = chain_test_results[chain_test_results_acc_map[key]]
    try:
      step = None
      test_loss = None
      train_loss = None

      string_split_apos = string.split(" - ")
      test_loss = float(string_split_apos[0].strip())

      result_string_split = string_split_apos[1].split("_")
      step = int(result_string_split[2].split(".p")[0].strip())
      train_loss = float(result_string_split[1].strip())*100
      
      indices.append(step)
      test_losses.append(test_loss)
      train_losses.append(train_loss)
    except Exception as e:
      print("[WARNING] Test - Encountered an exception while parsing string " + str(string) + ":")
      print(e)

  # We now have everything in our arrays. Combine them to graph 
  # them.   
  data= []
  for i in range(0, len(indices)):
    data.append([indices[i], test_losses[i], train_losses[i]])

  df = pd.DataFrame(data, columns = ["step", "test_loss", "train_loss"])

  # Sort the dataframe by step first so lines are drawn properly.
  df.sort_values("step", axis=0, inplace=True)

  df.set_index("step", drop=True, inplace=True)
  df = df.astype(float)

  # With our dataframe, we can finally graph the history. 
  plt.plot(df["train_loss"])
  plt.plot(df["test_loss"])
  plt.ylabel("Loss")
  plt.xlabel("Step")
  plt.legend(["train_loss", "test_loss"], loc="upper left")

  # Save the graph. 
  try:
    fig.savefig(graph_location)
    print("[DEBUG] Test - Graph successfully saved to: " + str(graph_location) + ".")
  except Exception as e:
    print("[ERROR] Test - Unable to save graph at location: " + str(graph_location) + ". Exception:")
    print(e)
  
  plt.close("all")