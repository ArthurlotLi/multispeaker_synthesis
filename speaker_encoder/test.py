#
# test.py
#
# Allows for the batch testing of all data inside of the preprocessed
# test data folder, given a directory of model(s). Saves a test
# report and a conclusive training history graph showing the train
# eer relative to the test eer. 

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import multiprocessing
import os

_model_suffix = ".pt"

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
def batch_test(model_batch_dir: Path, test_report_dir: Path, minibatch_size: int):
  print("[INFO] Test - Initializing batch test.")

  filename_uid = 0
  chain_test_results = {}
  chain_test_results_acc_map = {}

  # TODO: Create the dataloader now. 

  # Gathers all of the models in the directory. 
  model_minibatch = []
  files = os.listdir(model_batch_dir)

  for i in range(0, len(files)):
    filename = files[i]

    # Given a file, append it to the minibatch to be processed.
    if filename.endswith(_model_suffix):
      model_minibatch.append(filename)

    # If the minibatch has been filed OR there are no other files left.
    if (len(model_minibatch) == minibatch_size) or (i == len(files)-1):
      # Process the minibatch. 
      try:
        print("[INFO] Test - Processing minibatch: " + str(minibatch))
        # Dict object for the results of all tests. 
        ret_dict = {}
        queue = multiprocessing.Queue()
        queue.put(ret_dict)

        # Map of minibatch ids to tuples: (<Process>, <filename>)
        minibatch_processes = {}

        # Create a process for each model. 
        for j in range(0, len(model_minibatch)):
          file = minibatch[j]
          print("[INFO] Test - Creating new subprocess for model: %s." % file)

          # Execute the test as a separate process.
          # TODO: p = multiprocessing.Process()
          minibatch_processes[i] = (p, file)
        
        # After processes have been created, kick them off in parallel.
        for item in minibatch_processes:
          tuple = minibatch_processes[item]
          print("\n\n[INFO] Executing new process for model %s.\n" % tuple[1])
          tuple[0].start()
        
        # Now wait for all processes to finish. 
        for item in minibatch_processes:
          tuple = minibatch_processes[item]
          tuple[0].join()
        
        # All processes have now completed. Append the results to our
        # results dicts. 
        ret_dict_result = queue.get()
        print("\n[INFO] Processes complete; results:")
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
      minibatch = []
  
  if(filename_uid == 0):
    print("[WARNING] No files with suffix %s found at location \"%s\". Please specify another location with an argument or move/copy the model(s) accordingly." % (_model_suffix,model_batch_dir))
    return

  # Sort the results. 
  chain_test_results_acc_map = dict(sorted(chain_test_results_acc_map.items(), key=lambda item: item[0]))

  # With the results, write to file. 
  _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map) 

def _write_batch_results(test_report_dir, chain_test_results, chain_test_results_acc_map):
  try:
    results_folder_contents = os.listdir(test_report_dir)
    result_index = 0
    file_name_prefix = "chain_test_results_"
    file_name_suffix = ".txt"
    for file in results_folder_contents:
      file_number_str = file.replace(file_name_prefix, "").replace(file_name_suffix, "")
      file_number = -1
      try:
        file_number = int(file_number_str)
        if(file_number >= result_index):
          result_index = file_number + 1
      except:
        print("[WARN] Unexpected file in results directory. Ignoring...")

    filename = results_folder_contents + "/"+file_name_prefix+str(result_index)+file_name_suffix
    f = open(filename, "w")
    print("\n[INFO] Chain test complete. Writing results to file '"+filename+"'...")

    f.write("=================================\nSORTED Chain Test Results\n=================================\n\n")
    # Write results of each model, sorted.
    for key in chain_test_results_acc_map:
      f.write(chain_test_results[chain_test_results_acc_map[key]])

    f.close()
  except:
    print("[ERROR] Failed to write results to file!")
  
  _graph_train_test_history(test_report_dir, chain_test_results, chain_test_results_acc_map, filename)

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
def _graph_train_test_history(test_report_dir, chain_test_results, chain_test_results_acc_map, filename):
  print("[INFO] Generating test acc history graph.")

  # TODO: This function needs to be revamped to match the new format. NO VAL ACC! 

  graph_width_inches = 13
  graph_height_inches = 7
  
  graph_location = filename.replace(".txt", "")

  title = "Multispeaker Synthesis - Train + Test History"
  fig = plt.figure(1)
  fig.suptitle(title)
  fig.set_size_inches(graph_width_inches,graph_height_inches)

  # Gather points. Every point should be indexed by step. 
  indices = []
  test_eers = []
  train_accs = []

  # Each item should be structured as such:
  # 99.69913960 - encoder_0.028125_0.472423_0211800.pt
  for key in chain_test_results_acc_map: 
    string = chain_test_results[chain_test_results_acc_map[key]]
    try:
      step = None
      test_eer = None
      train_acc = None

      string_split_apos = string.split(" - ")
      test_eer = float(string_split_apos[0].strip())

      result_string_split = string_split_apos[1].split("_")
      step = int(result_string_split[5].split(".h")[0].strip())
      train_acc = float(result_string_split[4].strip())*100
      
      indices.append(step)
      test_eers.append(test_eer)
      train_accs.append(train_acc)
    except Exception as e:
      print("[WARNING] Encountered an exception while parsing string " + str(string) + ":")
      print(e)

  # We now have everything in our arrays. Combine them to graph 
  # them.   
  data= []
  for i in range(0, len(indices)):
    data.append([indices[i], test_eers[i], train_accs[i]])

  df = pd.DataFrame(data, columns = ["step", "test_eer", "train_acc"])

  # Sort the dataframe by step first so lines are drawn properly.
  df.sort_values("step", axis=0, inplace=True)

  df.set_index("step", drop=True, inplace=True)
  df = df.astype(float)

  # With our dataframe, we can finally graph the history. 
  plt.plot(df["test_eer"])
  plt.plot(df["train_acc"])
  plt.ylabel("Accuracy")
  plt.xlabel("Step")
  plt.legend(["test_eer", "train_acc"], loc="upper left")

  # Save the graph. 
  try:
    fig.savefig(graph_location)
    print("[DEBUG] Graph successfully saved to: " + str(graph_location) + ".")
  except Exception as e:
    print("[ERROR] Unable to save graph at location: " + str(graph_location) + ". Exception:")
    print(e)
  
  plt.close("all")