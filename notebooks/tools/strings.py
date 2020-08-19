from text_to_num import alpha2digit
import mmh3
import base64


def strip_prefix(prefix, string):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

def remove_suffix(word: str, suffix: str):
    """Remove a suffix from a string. """
    if word.endswith(suffix):
        return word[:-len(suffix)]
    return word

def convert_ordinal(word: str):
    """Convert a number to ordinal"""
    basic_forms = {"first": "one",
                   "second": "two",
                   "third": "three",
                   "fifth": "five",
                   "twelfth": "twelve"}
    
    for k, v in basic_forms.items():
        word = word.replace(k, v)
    
    word = word.replace("ieth", "y")
    
    for pattern in ["st", "nd", "rd", "th", "°"]:
        word = remove_suffix(word, pattern)
    
    converted = alpha2digit(word, "en")
    try:
        return int(converted)
    except:
        return None


def hash(word, pos):
    """
    Tool function to generate unique identifiers for the lexemes.
    """
    mmhash = mmh3.hash64(word + pos, signed=False)[0]
    mmhash = int.to_bytes(mmhash, 8, "big")
    return bytes.decode(base64.b32encode(mmhash)).rstrip("=").lower()
