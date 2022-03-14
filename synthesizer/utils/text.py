#
# text.py
#
# Text utilities - manages the functionality provided by the cleaners,
# numbers, and symbols utilities. 

from synthesizer.utils.symbols import symbols
import synthesizer.utils.cleaners
import re

# Mappings from symbol to numeric ID + vice versa.
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Text in curly braces.
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

# Converts a string of text to a sequence of IDs corresponding to
# the symbols in the text. Optionally, the text may contain ARPAbet
# sequences enclosed in curly braces.
# Ex) "Turn left on {HH AW1 S S T AH0 N} Street"
#
# Expects text to be converted into a sequence + cleaner name(s). 
# Returns the list of integers corresponding to the symbols in the
# text. 
def text_to_sequence(text, cleaner_names):
  sequence = []

  # Process any curly braces, treating their contents as ARPAbet.
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)
  
  # Append EOS token matching that defined in symbols.
  sequence.append(_symbol_to_id["~"])
  return sequence

# Given a sequence of token IDs, convert it back to a string.
def sequence_to_text(sequence):
  result = ""
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclosde ARPAbet back into curly braces.
      if len(s) > 1 and s[0] == "@":
        s = "{%s}" % s[1:]
      result += s
  return result.replace("}{", " ")

# Execute cleaners on text.
def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(synthesizer.utils.cleaners, name)
    if not cleaner:
      raise Exception("Unknown cleaner %s" % name)
    text = cleaner(text)
  return text

def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s not in ("_", "~")
