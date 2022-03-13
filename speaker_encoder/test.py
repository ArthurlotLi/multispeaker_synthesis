#
# test.py
#
# Allows for the batch testing of all data inside of the preprocessed
# test data folder, given a directory of model(s). Saves a test
# report and a conclusive training history graph showing the train
# eer relative to the test eer. 
#
# Works best if the test dataset is relatively small, as we load
# the entire dataset into memory for each subprocess rather than
# loading it batch by batch like during training. 

from speaker_encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataLoaderSequential, SpeakerVerificationDatasetSequential
from speaker_encoder.model import SpeakerEncoder

from pathlib import Path
from speaker_encoder.model_params import *
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch
import os

# How many to test at once! Limited by GPU VRAM
# NOTE: Multiprocessing is not necessary. The only thing this now 
#       affects is how often the tqdm updates.
_minibatch_size = 32

_projection_amount = 128 # Max amount of speakers to show in projection.
_model_suffix = ".pt"
_file_name_prefix = "chain_test_results_"
_file_name_suffix = ".txt"

# These factors determine how the EER is calculated. 
# The authors of the github paper used 6 utterances per speaker.
# For the batch size, we will be loading all speakers into one batch
# so as to provide the most "challenging" EER environment. 
_test_utterances_per_speaker = 10

# Colors to use during projection. 
colormap = np.random.rand(_projection_amount,3)

# Principal function for executing batch model testing, given a
# path to a folder containing models. Will execute testing with all
# files in this folder with the .pt suffix. Also requires the
# report directory.
#
# Expects each model file to be labeled as follows:
#   encoder_<eer>_<loss>_<step>.pt
#   Ex) encoder_0.028125_0.472423_0211800.pt
def batch_test(model_batch_dir: Path, clean_data_root: Path, 
               test_report_dir: Path, use_cpu: bool, projection: bool):
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
  model_minibatch = []
  files = os.listdir(model_batch_dir)
  files_models = []

  # Throw all applicable files into a list.
  for i in range(0, len(files)):
    filename = files[i]

    if filename.endswith(_model_suffix):
      files_models.append(filename)

  print("[INFO] Test - Found %d models in directory." % len(files_models))

  # We will load ALL data at once and compare all speakers in
  # the embedding to each other. 
  clean_data_contents = os.listdir(clean_data_root)
  num_speakers = 0
  for item in clean_data_contents:
    if clean_data_root.joinpath(item).is_dir():
      num_speakers += 1

  print("[INFO] Located %d speakers in %s." % (num_speakers, str(clean_data_root)))
  test_speakers_per_batch = num_speakers

  # Create the dataset + dataloader objects. 
  dataset = SpeakerVerificationDatasetSequential(clean_data_root)
  loader = SpeakerVerificationDataLoaderSequential(
    dataset,
    test_speakers_per_batch,
    _test_utterances_per_speaker,
    num_workers = data_loader_num_workers,
  )

  # Setup the device on which to run forward pass + loss calculations.
  # Note that these CAN be different devices, as the forward pass is
  # faster on the GPU whereas the loss (depending on what
  # hyperparameters you chose) is often faster on the CPU.
  device = None
  if use_cpu:
    device = torch.device("cpu")
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  loss_device = device

  print("[INFO] Test - Loading all %d speakers into memory..." % test_speakers_per_batch)

  #for step, speaker_batch in tqdm(enumerate(loader), desc="Testing Progress", total=len(loader)):
  for step, speaker_batch in enumerate(loader):
    print("\n[INFO] Starting step %d/%d\n" % (step, len(loader)))
    # Work through each model. 
    for i, model in tqdm(enumerate(files_models), desc="Models Tested", total=len(files_models)):
      # Given a file, append it to the minibatch to be processed.
      model_minibatch.append(model)

      # If the minibatch has been filed OR there are no other files left.
      if (len(model_minibatch) == _minibatch_size) or (i == len(files_models)-1):
        ret_dict = {}
        minibatch_processes = {}
        for j in range(0, len(model_minibatch)):
          file = model_minibatch[j]
          #print("\n[INFO] Test - Executing model: %s." % file)

          try:
            eer = _test_model_worker(model_batch_dir + "/" + file,
              test_report_dir,
              speaker_batch,
              result_index,
              projection,
              device,
              loss_device,
              test_speakers_per_batch)

            ret_dict["eer" + str(j)] = eer
            minibatch_processes[j] = (None, file)
          except Exception as e:
            print("\n\n[WARNING] Test - Encountered exception while testing file: %s" % file)
            print(e, end="\n\n")

        ret_dict_result = ret_dict
        #print("[DEBUG] Test - ret_dict_result: " + str(ret_dict_result))

        for item in ret_dict_result:
          item_identifier = int(item.replace("eer",""))
          if item_identifier in minibatch_processes:
            filename = minibatch_processes[int(item.replace("eer",""))][1]
            acc = ret_dict_result[item]
          
            if acc is None:
              print("[WARN] Test - Received empty eer!")
              chain_test_results[filename_uid] = "00.00000000 - " + str(filename) + " TEST FAILED!\n"
              chain_test_results_acc_map[-1] = filename_uid
            else:
              chain_test_results[filename_uid] = "%.8f - " % (acc*100) + str(filename) + "\n"
              # If a model of that exact accuracy exists already, append
              # a tiny number to it until it's unique. 
              if acc in chain_test_results_acc_map:
                sorting_acc = None
                while sorting_acc is None:
                  acc = acc + 0.000000000000001 # Acc has 15 decimal precision. Append by 1 to break ties.
                  if acc not in chain_test_results_acc_map:
                    sorting_acc = acc
                chain_test_results_acc_map[sorting_acc] = filename_uid
              else:
                chain_test_results_acc_map[acc] = filename_uid
            
            filename_uid = filename_uid + 1

        # Clear the minibatch regardless of success.   
        model_minibatch = []
  
  if(filename_uid == 0):
    print("[WARNING] Test - No files with suffix %s found at location \"%s\". Please specify another location with an argument or move/copy the model(s) accordingly." % (_model_suffix,model_batch_dir))
    return

  # Sort the results. 
  chain_test_results_acc_map = dict(sorted(chain_test_results_acc_map.items(), key=lambda item: item[0]))

  # With the results, write to file. 
  _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map, result_index) 

# Evaluate a model, given the location of the model, the dataloader,
# a queue object to insert the results into, and the key to use for
# said resuts. Intended to be run as a subprocess alongside parallel
# tests. 
def _test_model_worker(model_location, test_report_dir, speaker_batch, 
                       result_index, projection, device, loss_device, 
                       test_speakers_per_batch):
  # Load the model with the device.
  model = SpeakerEncoder(device, loss_device)
  checkpoint = torch.load(model_location, device)

  # To address some non-standard checkpoints.
  if "model_state" not in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
  else:  
    model.load_state_dict(checkpoint["model_state"], strict=False)

  # Always set the model into evaluation/inference mode.
  model.eval()

  #print("[INFO] Test - Loaded speaker encoder \"%s\" model trained to step %d." % (model_location, checkpoint["step"]))

  # Forward propagation. 
  inputs = torch.from_numpy(speaker_batch.data).to(device)
  embeds = model(inputs)

  # Calculate the loss with the loss device (Our CPU, not GPU) and
  # apply it. 
  embeds_loss = embeds.view((test_speakers_per_batch, _test_utterances_per_speaker, -1)).to(loss_device)
  loss, eer = model.loss(embeds_loss)
  
  if projection:
    #print("[DEBUG] Test - Drawing and saving projections.")
    filename = _file_name_prefix + str(result_index) + "_" + os.path.basename(str(model_location)).split(".p")[0] + ".png"
    projection_fpath = test_report_dir / filename
    # Visualize the embeddings.
    embeds = embeds.detach().cpu().numpy()
    _draw_projections(embeds, _test_utterances_per_speaker, projection_fpath)

  # Return the result.
  return eer

def _draw_projections(embeds, utterances_per_speaker, out_fpath):
  graph_width_inches = 13
  graph_height_inches = 7

  title = "Multispeaker Synthesis - UMAP Projection"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)

  max_speakers = len(colormap)
  embeds = embeds[:max_speakers * utterances_per_speaker]

  n_speakers = len(embeds) // utterances_per_speaker
  ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
  colors = [colormap[i] for i in ground_truth]

  reducer = umap.UMAP()
  projected = reducer.fit_transform(embeds)
  plt.scatter(projected[:, 0], projected[:, 1], c=colors)
  plt.gca().set_aspect("equal", "datalim")
  if out_fpath is not None:
    plt.savefig(out_fpath)
  plt.clf()

  plt.close("all")

# Writes results of batch testing to file. 
def _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map, result_index):
  try:
    filename = test_report_dir.joinpath(_file_name_prefix + str(result_index) + _file_name_suffix)
    print("\n[INFO] Test - Chain test complete. Writing results to file '%s'..." % filename)
    f = open(filename, "w")

    f.write("=================================\nSORTED Chain Test Results\n=================================\n\n")
    # Write results of each model, sorted.
    for key in chain_test_results_acc_map:
      f.write(chain_test_results[chain_test_results_acc_map[key]])

    f.close()
  except Exception as e:
    print("[ERROR] Failed to write results to file! Exception:")
    print(e)
  
  _graph_train_test_history(chain_test_results, chain_test_results_acc_map, filename)

  print("[INFO] Test - Write complete. Have a good night...")

# Given all the test accuracies that we've generated, let's 
# graph every single one. Note that we expect a very specific
# file name structure for these model iterations:
#
# Expects each model file to be labeled as follows:
#   encoder_<eer>_<loss>_<step>.pt
#   Ex) encoder_0.028125_0.472423_0211800.pt
#
# Expects full location of the file that has been written -
# the suffix will be stripped but otherwise the graph will be
# written to file with that exact name. 
def _graph_train_test_history(chain_test_results, chain_test_results_acc_map, filename):
  print("[INFO] Test - Generating test acc history graph.")

  graph_width_inches = 13
  graph_height_inches = 7
  
  graph_location = str(filename).replace(".txt", "")

  title = "Multispeaker Synthesis - Train + Test History"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)

  # Gather points. Every point should be indexed by step. 
  indices = []
  test_eers = []
  train_eers = []

  # Each item should be structured as such:
  # 99.69913960 - encoder_0.028125_0.472423_0211800.pt
  for key in chain_test_results_acc_map: 
    string = chain_test_results[chain_test_results_acc_map[key]]
    try:
      step = None
      test_eer = None
      train_eer = None

      string_split_apos = string.split(" - ")
      test_eer = float(string_split_apos[0].strip())

      result_string_split = string_split_apos[1].split("_")
      step = int(result_string_split[3].split(".p")[0].strip())
      train_eer = float(result_string_split[1].strip())*100
      
      indices.append(step)
      test_eers.append(test_eer)
      train_eers.append(train_eer)
    except Exception as e:
      print("[WARNING] Test - Encountered an exception while parsing string " + str(string) + ":")
      print(e)

  # We now have everything in our arrays. Combine them to graph 
  # them.   
  data= []
  for i in range(0, len(indices)):
    data.append([indices[i], test_eers[i], train_eers[i]])

  df = pd.DataFrame(data, columns = ["step", "test_eer", "train_eer"])

  # Sort the dataframe by step first so lines are drawn properly.
  df.sort_values("step", axis=0, inplace=True)

  df.set_index("step", drop=True, inplace=True)
  df = df.astype(float)

  # With our dataframe, we can finally graph the history. 
  plt.plot(df["train_eer"])
  plt.plot(df["test_eer"])
  plt.ylabel("EER")
  plt.xlabel("Step")
  plt.legend(["train_eer", "test_eer"], loc="upper left")

  # Save the graph. 
  try:
    fig.savefig(graph_location)
    print("[DEBUG] Test - Graph successfully saved to: " + str(graph_location) + ".")
  except Exception as e:
    print("[ERROR] Test - Unable to save graph at location: " + str(graph_location) + ". Exception:")
    print(e)
  
  plt.close("all")