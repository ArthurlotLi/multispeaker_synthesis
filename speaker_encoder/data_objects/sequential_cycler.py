#
# sequential_cycler.py
#
# Creates an internal source of a sequence and allows access to its
# items in a sequential order. Intended to allow repeatible sampling.

class SequentialCycler:
  def __init__(self, source):
    if len(source) == 0:
      raise Exception("SequentialCycler was provided an empty collection.")
    self.all_items = list(source)
    self.index = 0

  def sample(self, count: int):
    out = []
    while count > 0:
      if self.index >= len(self.all_items)-1:
        # Reset the index and start over. 
        self.index = 0 
      else:
        out.append(self.all_items[self.index])
        self.index += 1
        count -= 1
    return out

  def __next__(self):
    return self.sample(1)[0]