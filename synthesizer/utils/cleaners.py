#
# cleaners.py
#
# Cleaners - transformations that run over the input text before both
# training + inference. 
#
# These may be selected by passing a comma-delimited list of cleaner
# names in the "cleaner" hyperparameter (default: english_cleaners). 
#
# Some choices:
#   1. "english_cleaners" for English text
#   2. "transliteration_cleaners" for non-english text transliterated
#      to ASCII using the "Unidecode" python library
#   3. "basic_cleaners" if you do not want to transliterate. If you
#      use this one, you'll want to update the symbosl in symbol.py
#      to match your data. 

import re
from unidecode import unidecode
from synthesizer.utils.numbers import normalize_numbers

# All whitespace
_whitespace_re = re.compile(r"\s+")

# Replacing various abbreviations in english. 
_abbreviations = [(re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1]) for x in [
  ("mrs", "misess"),
  ("mr", "mister"),
  ("dr", "doctor"),
  ("st", "saint"),
  ("co", "company"),
  ("jr", "junior"),
  ("maj", "major"),
  ("gen", "general"),
  ("drs", "doctors"),
  ("rev", "reverend"),
  ("lt", "lieutenant"),
  ("hon", "honorable"),
  ("sgt", "sergeant"),
  ("capt", "captain"),
  ("esq", "esquire"),
  ("ltd", "limited"),
  ("col", "colonel"),
  ("ft", "fort"),
]]

# Apply the abbeviations dict
def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

# Apply numbers.py functionality
def expand_numbers(text):
  return normalize_numbers(text)

# Applied to all incoming text.
def lowercase(text):
  return text.lower()

# Normalize whitespace, so >=double-spaces turn into singular spaces.
def collapse_whitespace(text):
  return re.sub(_whitespace_re, " ", text)

def convert_to_ascii(text):
  return unidecode(text)

# Basic pipeline that lowercases + collapses whitespace without 
# transliteration.
def basic_cleaners(text):
  text = lowercase(text)
  text = collapse_whitespace
  return text

# Non-english text that transliterates to ASCII.
def transliteration_cleaners(text):
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

# English pipeline, including number + abbreviation expansion.
def english_cleaners(text):
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text