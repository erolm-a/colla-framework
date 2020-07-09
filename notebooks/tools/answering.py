"""A bunch of QA functions"""

import regex as re
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np
import pandas as pd
from text_to_num import alpha2digit

from .providers import WiktionaryProvider

model = SentenceTransformer('bert-base-nli-mean-tokens')

class DefinitionIntent:
    """Intent for a lookup definition. It is used to map simple questions like
    "define chihuaua"."""
    def __init__(self, noun_phrase):
        self._noun_phrase = noun_phrase
    
    @property
    def noun_phrase(self):
        return self._noun_phrase


class DefinitionEntity:
    """Fetched entity data obtained from a DefinitionIntent. This entity is meant
    to be used in conjunction with FilterIntents """
    def __init__(self, noun_phrase, senses):
        self._noun_phrase = noun_phrase
        self._senses = senses
    
    @property
    def noun_phrase(self):
        return self._noun_phrase

    @property
    def senses(self):
        return self._senses


class FilterIntent:
    def __init__(self, filter_type, filter_values):
        """Filter some results from a DefinitionEntity.
        
        Possible values for `filter_type` are:
        - single

        if filter_type is:
            - "single", then filter_values is the number 
        """
        self._filter_type = filter_type
        self._filter_values = filter_values
    
    @property
    def filter_type(self):
        return self._filter_type
    
    @property
    def involving(self):
        if self.filter_type == "single":
            return self._filter_values


class QuestionAnsweringContext:
    """Context for a chatbot."""
    def __init__(self):
        self.entities = None
        pass

    def handle_intent(self, intent):
        if isinstance(intent, FilterIntent):
            if intent.filter_type == "single":
                number = intent.involving
                if self.entities:
                    return self.entities.senses[number]
                
        if isinstance(intent, DefinitionIntent):
            noun_phrase = intent.noun_phrase
            self.entities = DefinitionEntity(noun_phrase, find_exact_entities(noun_phrase))
            return self.entities
    
    def handle_question(self, question):
        intent = match_intent_question(question)
        if intent is not None:
            return self.handle_intent(intent)
        else:
            return {'response': 'unable to handle this question'}, 401

basic_definition_regex = re.compile(r'(?:define|definition of|say|how do you define)? ?"?(?P<NP>"?.*)"?')
filter_entity_singular_regex = re.compile(r'(?:give me the)|(?:tell me the )|(?:show me the)(?:the )|(?:)(?P<number>.+)')

def match_intent_question(question: str):
    # TODO: use a fully-fledged CFG or a language model for this task
    question = question.strip()
    question.replace("'", '"')
    
    matches = basic_definition_regex.match(question)
    if matches:
        noun_phrase = matches.group("NP")
        return DefinitionIntent(noun_phrase)

    matches = filter_entity_singular_regex.match(alpha2digit(question, lang="en"))
    if matches:
        number = matches.group("number")
        number_match = re.match(r'(^\d+)', number)
        if number_match:
            return FilterIntent('single', int(number_match.group(0)))


def answer_natural_question(question: str):
    # strip whitespace before and after
    # Completely ignore the NLP pipeline by now (tokenization etc.)
    intent = match_intent_question(question)
    if isinstance(intent, DefinitionIntent):
        noun_phrase = intent.noun_phrase
        return find_exact_entities(noun_phrase)
    else:
        raise Exception(": " + question)


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