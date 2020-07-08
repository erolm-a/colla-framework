"""A bunch of QA functions"""

import re
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np
import pandas as pd

from .providers import WiktionaryProvider

model = SentenceTransformer('bert-base-nli-mean-tokens')

basic_definition_regex = re.compile(r'(?:define|definition of|say|how do you define)? ?"?(?P<NP>"?.*)"?')

def parse_natural_question(question: str):
    # strip whitespace before and after
    # Completely ignore the NLP pipeline by now (tokenization etc.)
    question = question.strip()
    question.replace("'", '"')
    
    matches = basic_definition_regex.match(question)
    if len(matches.groups()) == 0:
        raise Exception("Unable to parse this sentence")

    noun_phrase = basic_definition_regex.match(question).group("NP")
    return find_exact_entities(noun_phrase)


def find_exact_entities(label: str):
    # Just wiktionary results
    results_df = WiktionaryProvider().fetch_by_label(label, format="json")
    results_marshalling = []
    for row in results_df.iterrows():
        item = row[1]
        results_marshalling.append({'pos': item["partOfSpeech"].lower(),
                                    'definition': item["strippedDefinition"],
                                    'examples': item["examples"],
                                    'related': []})
    return results_marshalling

    