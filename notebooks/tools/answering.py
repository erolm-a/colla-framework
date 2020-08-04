"""A bunch of QA functions"""

from enum import Enum
from typing import List
from sentence_transformers import SentenceTransformer
# from scipy.spatial import distance
import numpy as np
import pandas as pd
from random import randint

from .providers import FusekiProvider
from .grammar import grammar, pick_best_semantics, print_parse

from .globals import logger

from .strings import convert_ordinal, remove_suffix, strip_prefix

provider = FusekiProvider("sample_10000_common")

model = SentenceTransformer('bert-base-nli-mean-tokens')

class SerializedIntent:
    """This class serves as the output of an intent matcher that should then be serialized via Flask to the user."""
    class IntentType(Enum):
        WELCOME = "welcomeIntent" # technically not an intent
        DEFINITION = "definitionIntent"
        FILTER = "filterIntent"
        ERROR = "errorIntent"
    
    def __init__(self, intentType: IntentType, message: str):
        self.intentType = intentType
        self.message = message

def failed_intent(error_message: str) -> SerializedIntent:
    return SerializedIntent(SerializedIntent.IntentType.ERROR, error_message)

class Sense:
    """A mere wrapper for senses"""
    def __init__(self, id: str, description: str, pos: str,
                 examples: List[str], related: List[str]):
        self.id = id
        self.description = description
        self.pos = pos
        self.examples = examples
        self.related = related

class Intent:
    """Base class for all the intents"""
    pass

class DefinitionIntent(Intent):
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
    def __init__(self, noun_phrase: str, senses: List[Sense]):
        self._noun_phrase = noun_phrase
        self._senses = senses
    
    def serialize(self) -> SerializedIntent:
        """Serialize the current state of the definitions into a human-readable output"""
        logger.debug("Serializing an answer here...")
        message = """Found the following meanings"""
        if len(self.senses) > 5:
            message += " (limited to 5 senses, unless you want more)"
        
        message += "<ul>"
        for sense in self.senses[:5]:
            message += f"<li>({sense.pos}) {sense.description}</li>"
        message += "</ul>"

        return SerializedIntent(SerializedIntent.IntentType.DEFINITION, message)

    @property
    def noun_phrase(self):
        return self._noun_phrase

    @property
    def senses(self) -> List[Sense]:
        return self._senses


class FilterIntent(Intent):
    def __init__(self, filter_type, filter_values):
        """Filter some results from a DefinitionEntity.
        
        Possible values for `filter_type` are:
        - single
        - sensical

        if filter_type is:
            - "single", then filter_values is the number
            - "sensical", then use BERT to determine which given sense is desired.
        """
        self._filter_type = filter_type
        self._filter_values = filter_values
    
    @property
    def filter_type(self):
        return self._filter_type
    
    @property
    def involving(self):
        if self.filter_type in ["single", "sensical"]:
            return self._filter_values



class QuestionAnsweringContext:
    """Context for a chatbot."""
    def __init__(self):
        self.entities = None
        pass

    def handle_intent(self, intent: Intent) -> SerializedIntent:
        logger.debug("Current state of the entities: " + repr(self.entities))
        if isinstance(intent, FilterIntent):
            if intent.filter_type == "single":
                number = intent.involving
                if self.entities:
                    max_range = len(self.entities.senses)
                    if number not in range(0, max_range):
                        return failed_intent(f"This sense does not exist! Try asking me to fetch the {randint(1, max_range)})°")
                    sense = self.entities.senses[number - 1]
                    message = f"More details on the item no. {number}: <br>"
                    message += f"<i>({sense.pos})</i> {sense.description}"

                    if len(sense.examples) > 0:
                        message += "<br>Examples:<br>"
                        message += "<br>".join(sense.examples)

                    if len(sense.related) > 0:
                        message += "<br>Examples:<br>"
                        message += "<br>".join(sense.related)
                    
                    return SerializedIntent(SerializedIntent.IntentType.FILTER, message)
                
                # No previous DefinitionEntity here
                else:
                    return SerializedIntent(SerializedIntent.IntentType.ERROR, "I need a definition to work on!"), 400
            
            # Sensical
            elif intent.filter_type == "sensical":
                most_related = find_most_related_entity(self.entities, intent.involving)

                message = "I think you mean the following:<br>"
                for sense in self.entities.senses:
                    if sense.id == most_related[0]:
                        return SerializedIntent(SerializedIntent.IntentType.FILTER, message + sense.description)
                

                
        if isinstance(intent, DefinitionIntent):
            noun_phrase = intent.noun_phrase
            self.entities = find_exact_entities(noun_phrase)

            return self.entities.serialize()
    
    def handle_question(self, question) -> SerializedIntent:
        logger.info(f"Obtained question: {question}")
        intent = match_intent_question(question)
        if intent is not None:
            return self.handle_intent(intent)
        else:
            return failed_intent('unable to handle this question'), 401


def match_intent_question(question: str) -> Intent:
    # TODO: use a fully-fledged CFG or a language model for this task
    question = question.strip().lower()
    question.replace("'", '"')

    for eos in [".", "?", "!"]:
        question = remove_suffix(question, eos)

    print_parse([question])
    parses = grammar.parse_input(question)
    best_semantics = pick_best_semantics(parses)
    logger.debug(repr(best_semantics))

    if best_semantics['intent'] == 'definition':
        return DefinitionIntent(best_semantics['np'])
    elif best_semantics['intent'] == 'filter':
        if best_semantics['type'] == "number":
            return FilterIntent('single', best_semantics['value'])
        elif best_semantics['type'] == "sense_meaning":
            return FilterIntent('sensical', best_semantics['value'])
    


def find_exact_entities(label: str) -> DefinitionEntity:
    # Just wiktionary results
    results_df = provider.fetch_by_label(label, format="json")
    results_marshalling = []
    for row in results_df.iterrows():
        item = row[1]
        results_marshalling.append(Sense(item["sense"],
                                              item["senseDefinition"],
                                              strip_prefix("kgl:", item["pos"].lower()),
                                              [],
                                              []))
    return DefinitionEntity(label, results_marshalling)


def find_most_related_entity(definition_entity: DefinitionEntity, descriptions):
    """
    Find a list of related entities that match the given descriptions.
    This method uses a language model to disambiguate between different results.
    
    descriptions can be either a single string or an iterable of strings. In the latter case, the function
    will return a list of matching entities with the given order.
    """
    descriptions_as_list = not isinstance(descriptions, str)
    
    if not descriptions_as_list:
        descriptions = [descriptions]
    
    definitions = [sense.description for sense in definition_entity.senses] + descriptions
    definitions = np.array(definitions)
    
    # Get S-BERT
    embeddings = model.encode(definitions)
    embeddings_np = np.array((embeddings))
    embeddings_normalized = embeddings_np / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)    
    embedding_results, embedding_queries = embeddings_normalized[:-len(descriptions)] , embeddings_normalized[-len(descriptions):]
    
    # perform cosine similarity between all the queries and all the given descriptions
    # I can't find a way to make this vectorized, but should not be a concern.
    correlation_scores = [query.dot(embedding_results.T) for query in embedding_queries]
    correlation_scores = np.stack(correlation_scores)
    
    # Match the queries with the entities based on maximum cosine similarity
    preferred = np.argmax(correlation_scores, axis=1)
    result =  [definition_entity.senses[idx].id for idx in preferred]
    
    return result 
