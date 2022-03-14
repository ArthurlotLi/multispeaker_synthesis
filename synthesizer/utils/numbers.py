#
# numbers.py
#
# Working with numbers in strings. For preprocessing, always expand
# numbers to textual counterparts. 

import re
import inflect

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

# Main function to use. 
def normalize_numbers(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r"\1 pounds", text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text

# We don't include commas.
def _remove_commas(m):
  return m.group(1).replace(",", "")

# Split the string with a " point " for phonetic pronunciation.
def _expand_decimal_point(m):
  return m.group(1).replace(".", " point ")

# Given dollars, expand into a phrase. 
def _expand_dollars(m):
  match = m.group(1)
  parts = match.split(".")
  if len(parts) > 2:
    return match + " dollars" # Unexpected format. Return.
  
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = "dollar" if dollars == 1 else "dollars"
    cent_unit = "cent" if cents == 1 else "cents"
    return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = "dollar" if dollars == 1 else "dollars"
    return "%s %s" % (dollars, dollar_unit)
  elif cents:
    cent_unit = "cent" if cents == 1 else "cents"
    return "%s %s" % (cents, cent_unit)
  else:
    return "zero dollars"

# Let inflect do the heavy lifting.
def _expand_ordinal(m):
  return _inflect.number_to_words(m.group(0))

# Handle numbers specially for years. 
def _expand_number(m):
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return "two thousand"
    elif num > 2000 and num < 2010:
      return "two thousand " + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
  else:
    return _inflect.number_to_words(num, andword = "")