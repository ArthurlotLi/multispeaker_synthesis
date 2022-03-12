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
#from utils.profiler import Profiler

from pathlib import Path
from speaker_encoder.model_params import *
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import torch
import multiprocessing
import os

_model_suffix = ".pt"
_test_batch_size = 512 # Load the entire dataset at once. 
_file_name_prefix = "chain_test_results_"
_file_name_suffix = ".txt"

# A set of colors to use. Subjective.
colormap = np.random.rand(128,3)
"""
colormap = np.array([
  [76, 255, 0],
  [0, 127, 70],
  [255, 0, 0],
  [255, 217, 38],
  [0, 135, 255],
  [165, 0, 165],
  [255, 167, 255],
  [0, 255, 255],
  [255, 96, 38],
  [142, 76, 0],
  [33, 0, 127],
  [0, 0, 0],
  [183, 183, 183],
], dtype=np.float) / 255
"""

# As cuda operations are async, synchronize for correct profiling.
def sync(device: torch.device):
  if device.type == "cuda":
    torch.cuda.synchronize(device)

# Principal function for executing batch model testing, given a
# path to a folder containing models. Will execute testing with all
# files in this folder with the .pt suffix. Also requires the
# report directory.
#
# Expects each model file to be labeled as follows:
#   encoder_<eer>_<loss>_<step>.pt
#   Ex) encoder_0.028125_0.472423_0211800.pt
def batch_test(model_batch_dir: Path, clean_data_root: Path, test_report_dir: Path, minibatch_size: int, use_cpu: bool):
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

  # Load the test dataset. We provide this to every subprocess. 
  # TODO: Implement
  #test_dataset = _load_test_set(clean_data_root)

  # Gathers all of the models in the directory. 
  model_minibatch = []
  files = os.listdir(model_batch_dir)
  files_models = []

  # Throw all applicable files into a list.
  for i in range(0, len(files)):
    filename = files[i]

    if filename.endswith(_model_suffix):
      files_models.append(filename)

  # Work through each model. 
  for model in files_models:
    # Given a file, append it to the minibatch to be processed.
    model_minibatch.append(model)

    # If the minibatch has been filed OR there are no other files left.
    if (len(model_minibatch) == minibatch_size) or (i == len(files)-1):
      # Process the minibatch. 
      try:
        print("[INFO] Test - Processing minibatch: " + str(model_minibatch))
        # Dict object for the results of all tests. 
        ret_dict = {}
        queue = multiprocessing.Queue()
        queue.put(ret_dict)

        # Map of minibatch ids to tuples: (<Process>, <filename>)
        minibatch_processes = {}

        # Create a process for each model. 
        for j in range(0, len(model_minibatch)):
          file = model_minibatch[j]
          print("[INFO] Test - Creating new subprocess for model: %s." % file)

          # Execute the test as a separate process.
          p = multiprocessing.Process(target=_test_model_worker, args=(
            model_batch_dir + "/" + file,
            test_report_dir,
            clean_data_root,
            queue,
            use_cpu,
            "eer" + str(i),
            result_index
          ))
          minibatch_processes[i] = (p, file)
        
        # After processes have been created, kick them off in parallel.
        for item in minibatch_processes:
          tuple = minibatch_processes[item]
          print("\n\n[INFO] Test - Executing new process for model %s.\n" % tuple[1])
          tuple[0].start()
        
        # Now wait for all processes to finish. 
        for item in minibatch_processes:
          tuple = minibatch_processes[item]
          tuple[0].join()
        
        # All processes have now completed. Append the results to our
        # results dicts. 
        ret_dict_result = queue.get()
        print("\n[INFO] Test - Processes complete; results:")
        print(ret_dict_result) 
        print("")

        for item in ret_dict_result:
          item_identifier = int(item.replace("eer",""))
          if item_identifier in minibatch_processes:
            filename = minibatch_processes[int(item.replace("eer",""))][1]
            acc = ret_dict_result[item]
          
            if acc is None:
              print("[WARN] Received empty eer!")
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

      except Exception as e:
        print("\n\n[ERROR] Test - FAILED to process minibatch " + str(model_minibatch) + " ! Exception: ")
        print(e)
        print("\n")

      # Clear the minibatch regardless of success.   
      model_minibatch = []
  
  if(filename_uid == 0):
    print("[WARNING] No files with suffix %s found at location \"%s\". Please specify another location with an argument or move/copy the model(s) accordingly." % (_model_suffix,model_batch_dir))
    return

  # Sort the results. 
  chain_test_results_acc_map = dict(sorted(chain_test_results_acc_map.items(), key=lambda item: item[0]))

  # With the results, write to file. 
  _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map, result_index) 

# Loads the test set into one combined .npy file.
# TODO: Implement more efficient loading. 
""" 
def _load_test_set(clean_data_root):
  test_set = None
  try:
    print("[DEBUG] Attempting to load test dataset at %s." % clean_data_root)
    test_set = 
  except Exception as e:
    print("[ERROR] Failed to read test set at %s. Exception:" % clean_data_root)
    print(e)
  return test_set
"""

# Evaluate a model, given the location of the model, the dataloader,
# a queue object to insert the results into, and the key to use for
# said resuts. Intended to be run as a subprocess alongside parallel
# tests. 
def _test_model_worker(model_location, test_report_dir, clean_data_root, queue, use_cpu, queue_key, result_index):
  # Create the dataset + dataloader objects. 
  dataset = SpeakerVerificationDatasetSequential(clean_data_root)
  loader = SpeakerVerificationDataLoaderSequential(
    dataset,
    _test_batch_size,
    utterances_per_speaker,
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

  # Load the model with the device.
  model = SpeakerEncoder(device, loss_device)
  checkpoint = torch.load(model_location, device)
  model.load_state_dict(checkpoint["model_state"], strict=False)

  # Always set the model into evaluation/inference mode.
  model.eval()

  print("[INFO] Test - Loaded speaker encoder \"%s\" model trained to step %d." % (model_location, checkpoint["step"]))

  total_eer = 0

  # Execute evaluation. 

  #for step, speaker_batch in enumerate(loader, 1):
  for step, speaker_batch in tqdm(enumerate(loader), desc="Testing Progress", total=len(loader)):
    # Forward propagation. 

    # Send the next batch to the device and propagate them through 
    # the network.
    inputs = torch.from_numpy(speaker_batch.data).to(device)
    sync(device)

    # Get the embedding vector output by the model. 
    embeds = model(inputs)
    sync(device)

    # Calculate the loss with the loss device (Our CPU, not GPU) and
    # apply it. 
    embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
    loss, eer = model.loss(embeds_loss)
    sync(loss_device)

    if total_eer != 0:
      total_eer = (total_eer + eer)/2
    else:
      total_eer = eer
    
    print("[DEBUG] Test - Drawing and saving projections.")
    filename = _file_name_prefix + str(result_index) + "_" + os.path.basename(str(model_location)).split(".p")[0] + ".png"
    projection_fpath = test_report_dir / filename
    # Visualize the embeddings.
    embeds = embeds.detach().cpu().numpy()
    _draw_projections(embeds, utterances_per_speaker, step, projection_fpath)

  # Return the result in the queue.
  ret_dict = queue.get()
  ret_dict[queue_key] = total_eer
  queue.put(ret_dict)

def _draw_projections(embeds, utterances_per_speaker, step, out_fpath):
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
  #try:

  filename = test_report_dir.joinpath(_file_name_prefix + str(result_index) + _file_name_suffix)
  print("\n[INFO] Chain test complete. Writing results to file '%s'..." % filename)
  f = open(filename, "w")

  f.write("=================================\nSORTED Chain Test Results\n=================================\n\n")
  # Write results of each model, sorted.
  for key in chain_test_results_acc_map:
    f.write(chain_test_results[chain_test_results_acc_map[key]])

  f.close()
  #except Exception as e:
    #print("[ERROR] Failed to write results to file! Exception:")
    #print(e)
  
  _graph_train_test_history(chain_test_results, chain_test_results_acc_map, filename)

  print("[INFO] Write complete. Have a good night...")

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
  print("[INFO] Generating test acc history graph.")

  # TODO: This function needs to be revamped to match the new format. NO VAL ACC! 

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
      print("[WARNING] Encountered an exception while parsing string " + str(string) + ":")
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
  plt.plot(df["test_eer"])
  plt.plot(df["train_eer"])
  plt.ylabel("Accuracy")
  plt.xlabel("Step")
  plt.legend(["test_eer", "train_eer"], loc="upper left")

  # Save the graph. 
  try:
    fig.savefig(graph_location)
    print("[DEBUG] Graph successfully saved to: " + str(graph_location) + ".")
  except Exception as e:
    print("[ERROR] Unable to save graph at location: " + str(graph_location) + ". Exception:")
    print(e)
  
  plt.close("all")