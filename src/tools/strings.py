"""
Generic string utils.
"""

import mmh3
import base64

from trie_search import RecordTrieSearch
from text_to_num import alpha2digit


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


class MyRecordTrie(RecordTrieSearch):
    """
    A variant of a RecordTrieSearch for the purpose of our autolinker.
    The main difference is that it assumes that the input text has already been tokenized
    """
    def __init__(self, records):
        super().__init__("<Q", records)

    def search_all_patterns(self, tokens):
        words = [tok[0] for tok in tokens]

        for i, (word, span) in enumerate(tokens):
            for pattern in self._TrieSearch__search_prefix_patterns(word, words[i+1:]):
                weight = self[pattern][0][0]
                yield pattern, span[0], weight, i # exact token position
    
    def search_longest_patterns(self, tokens):
        # avoid overlapping mentions
        all_patterns = self.search_all_patterns(tokens)
        check_field = [0] * len(tokens)
        for pattern, start_idx, weight, idx in sorted(
                all_patterns, key=lambda x: len(x[0]), reverse=True):
            pattern_length = pattern.count(" ") + 1
            target_field = check_field[idx:idx + pattern_length]
            check_sum = sum(target_field)
            if check_sum == 0:
                for i in range(pattern_length):
                    check_field[idx + i] = 1
                yield pattern, start_idx, weight, idx