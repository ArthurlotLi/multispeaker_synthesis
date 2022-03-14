#
# symbols.py
#
# Defines the set of symbols used in the text input to the model.
#
# The default is the set of ASCII characters that generally applies to
# English or text run through Unidecode. For other data (when using
# basic_cleaners), you can modify _characters. 

_pad        = "_"
_eos        = "~"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet