#
# m4a_to_wav.py
#
# Standalone utility that converts the contents of a directory from
# m4a files to wav files that are far more tractable for usage. 
# Preserves the directory structure of the original directory. 
# Requires FFmpeg in the system path. 
#
# Note the new directory cannot exist when you kick this off.
#
# Ex) python m4a_to_wav.py ../datasets/voxceleb2_aac ../datasets/VoxCeleb2

# TODO: If you ever use this file again, implement multiprocessing.

import argparse
import shutil
import os

#_original_format = ".m4a"
_original_format = ".mp3"

# Allows shutil to ingnore all files.
def ig_f(dir, files):
  return [f for f in files if (os.path.isfile(os.path.join(dir, f)))] #and not os.path.join(dir, f).endswith(".m4a"))]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("original_directory")
  parser.add_argument("new_directory")
  args = parser.parse_args()

  original_directory = args.original_directory
  new_directory = args.new_directory

  # Step 1: Create the new file tree. 
  print("[INFO] Creating new directory at: '%s'..." % new_directory)
  shutil.copytree(original_directory, new_directory, ignore=ig_f)

  # Step 2: Copy every single m4a file. 
  print("[INFO] Walking all files in the original directory.")
  m4a_files = 0
  non_m4a_files = 0
  for root, dirs, files in os.walk(original_directory, topdown=False):
    for name in files:
      # Given a file like "../datasets/test_aac/6gecvwZYs-o/00024.m4a"
      full_path = os.path.join(root, name)
      if not full_path.endswith(_original_format):
        print("[WARNING] Encountered non-%s file: %s. Skipping..." % (_original_format, full_path))
        non_m4a_files += 1
      else:
        if(original_directory in full_path):
          new_path = full_path.replace(original_directory, new_directory)
        else:
          print("[ERROR] Unable to find original directory in full path: %s\n\nStopping..." % full_path)
          break
        # In case we have a frankenstein of / and \. 
        full_path = os.path.normpath(full_path)
        new_path = os.path.normpath(new_path)
        new_path = new_path.replace(_original_format, ".wav")

        # Execute the conversion.
        print("Converting %s file: %s --> %s" % (_original_format, full_path, new_path))
        command = "ffmpeg -strict -2 -i \"" + full_path + "\" \"" + new_path + "\""
        os.system(command)
        m4a_files += 1
      

  # Step 3: Profit.
  print("\n[INFO] %s to wav file conversion complete! \n  %s files converted: %d \n  Non-%s files skipped: %d" 
    % (_original_format,_original_format, m4a_files, _original_format, non_m4a_files))