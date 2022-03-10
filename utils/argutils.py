#
# argutils.py
#
# Supplemental tools for helping output args to the user during
# preprocessing and training. 

from pathlib import Path
import numpy as np
import argparse

# Order to print things in, in decreasing order.
_type_priorities = [
  Path,
  str,
  int,
  float,
  bool
]

# Use the priorities to get the next priority item. 
def _priority(o):
  p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None)
  if p is not None: return p

  # Not found with type(); try with isinstance.
  p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None)
  if p is not None: return p

  # Otherwise, just return default number of types.
  return len(_type_priorities)

# Print all arguments. Allows you to pass in a parser that you've
# created and used.
def print_args(args: argparse.Namespace, parser=None):
  args = vars(args)
  if parser is None:
    priorities = list(map(_priority, args.values()))
  else:
    all_params = [a.dest for g in parser._action_groups for a in g._group_actions ]
    priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
    priorities = list(map(priority, args.keys()))

  pad = max(map(len, args.keys())) + 3
  indices = np.lexsort((list(args.keys()), priorities))
  items = list(args.items())
  
  print("Arguments:")
  for i in indices:
      param, value = items[i]
      print("    {0}:{1}{2}".format(param, ' ' * (pad - len(param)), value))
  print("")
    