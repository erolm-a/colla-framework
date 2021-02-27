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
    """
    A wrapper and endpoint for sense-based search.
    """
    def __init__(self, id: str, parent_lexeme: str, parent_lexeme_str: str,
                 description: str, pos: str, examples: List[str]):
        self.id = id
        self.parent_lexeme = parent_lexeme
        self.parent_lexeme_str = parent_lexeme_str
        self.description = description
        self.pos = pos
        self.examples = examples
    
    def show_examples(self):
        message = ""
        if len(self.examples) > 0:
            message = "<br>".join(self.examples)
        return message

    def show_usages(self):
        usages = provider.fetch_usages(self.id)
        if len(usages) > 0:
            message = "Usages: "
            message += "".join(["<br>" + usage for usage in usages['description.value'].to_list()])
        else:
            message = "I am not aware of usages for the given sense"
        return message

    def show_forms(self, required_form):
        # TODO
        pass


class QuestionAnsweringContext:
    """Context for a chatbot."""

    def __init__(self):
        self.entities = None
        pass

    def handle_question(self, question) -> SerializedIntent:
        logger.info(f"Obtained question: {question}")
        intent = match_intent_question(question)
        if intent is not None:
            return intent.handle_intent(self)
        else:
            return failed_intent('unable to handle this question'), 401


class Intent:
    """Base class for all the intents"""
    def handle_intent(self, context: QuestionAnsweringContext):
        pass


class DefinitionIntent(Intent):
    """Intent for a lookup definition. It is used to map simple questions like
    "define chihuaua"."""
    def __init__(self, noun_phrase):
        self._noun_phrase = noun_phrase
    
    @property
    def noun_phrase(self):
        return self._noun_phrase


    def handle_intent(self, context: QuestionAnsweringContext):
        context.entities = find_exact_entities(self.noun_phrase)
        if context.entities is None:
            return failed_intent("I could not find what you were looking for")
        return context.entities.serialize()


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
    def __init__(self, intent_dict):
        """Filter some results from a DefinitionEntity.

        Please refer to the Intent documentation.
        """
        self._parse(intent_dict)

    def _parse(self, intent_dict):
        self._filter_type = intent_dict.get("filtertype", None)
        if self.filter_type in ["number", "semantic"]:
            self._filter_values = intent_dict["value"]

        self._variant = intent_dict.get("variant", None)
        self.slots = intent_dict

    @property
    def filter_type(self):
        return self._filter_type
    
    @property
    def involving(self):
        if self.filter_type in ["number", "semantic"]:
            return self._filter_values
    
    @property
    def variant(self) -> str:
        return self._variant

    def handle_variant(self, senses: List[Sense]):
        message = ""
        for sense in senses:
            if self.variant is None:
                message += sense.show_examples()
                message += sense.show_related()

            elif self.variant == "example":
                message += sense.show_examples()

            elif self.variant == "usages":
                message += sense.show_usages()
        return message

    def handle_intent(self, context) -> SerializedIntent:
        message = ""

        if not context.entities or not context.entities.senses:
            return failed_intent("I don't have definitions to work on!")

        matching_senses = context.entities.senses
        if self.filter_type == "number":
            number = self.involving
            logger.info(number)

            max_range = len(context.entities.senses)
            logger.info(max_range)
            if number not in range(0, max_range):
                return failed_intent(f"This sense does not exist! Try asking me to tell you about the {randint(1, max_range)})°")

            matching_senses = [context.entities.senses[number - 1]]

            message += f"More details on the item no. {number}: <br>"
            if not self.variant:
                message += f"<i>({matching_senses[0].pos})</i> {matching_senses[0].description}"

        # Semantic
        elif self.filter_type == "semantic":
            related_match, confidence = find_most_related_entity(context.entities, self.involving)
            related_match = related_match[0]
            print(confidence)
            confidence = confidence[0]

            if confidence < 0.5:
                return failed_intent("I am not sure I got what you meant.")

            message = "I think you meant the following:<br>"
            matching_senses = [s for s in context.entities.senses if s.id == related_match]
            if not self.variant:
                message += matching_senses[0].description

        # Grammatical
        elif self.filter_type == "grammatical":
            required_pos = self.slots.get("requiredPos", None)
            if required_pos:
                matching_senses = [sense for sense in context.entities.senses if sense.pos == required_pos]
                if len(matching_senses) == 1:
                    sense = matching_senses[0]
                    if not self.variant:
                        message += sense.description
                        message += sense.show_examples()

            grammatical_feature = self.slots.get("grammaticalFeature", None)
            if grammatical_feature:
                distinct_senses = {}
                for sense in matching_senses:
                    if not sense.parent_lexeme_str in distinct_senses:
                        distinct_senses[sense.parent_lexeme_str] = sense

                for sense in distinct_senses.values():
                    forms = provider.fetch_forms(sense, grammatical_feature)
                    if len(forms) > 0:
                        message += f"Found the following forms for {sense.parent_lexeme_str} ({sense.pos})"
                        for row in forms["formLabel"].iterrows():
                            form = row[1]
                            message += f"<br>{form}"
                    else:
                        return failed_intent("I could not find matching forms")

        if self.variant:
            message += self.handle_variant(matching_senses)

        return SerializedIntent(SerializedIntent.IntentType.FILTER, message), 200

def match_intent_question(question: str) -> Intent:
    # TODO: taking the lowercase version of the string is not always a good solution.
    question = question.strip().lower()
    question.replace("'", '"')

    for eos in [".", "?", "!"]:
        question = remove_suffix(question, eos)

    print_parse([question])
    parses = grammar.parse_input(question)
    best_semantics = pick_best_semantics(parses)
    logger.debug(repr(best_semantics))

    # Failed to match this intent
    if not 'intent' in best_semantics:
        return

    if best_semantics['intent'] == 'define':
        return DefinitionIntent(best_semantics['np'])
    elif best_semantics['intent'] == 'filter':
        return FilterIntent(best_semantics)


def find_exact_entities(label: str) -> DefinitionEntity:
    # Just wiktionary results
    results_df = provider.fetch_by_label(label, format="json")
    if results_df is None:
        return None
    results_marshalling = []
    # TODO: the given label could (and *will*) be a conjugated noun.
    # In theory the full wiktionary includes entries for extracted forms, but
    # we shouldn't rely on that.
    for row in results_df.iterrows():
        item = row[1]
        results_marshalling.append(Sense(item["sense"],
                                              item["entity"],
                                              label,
                                              item["senseDefinition"],
                                              strip_prefix("kgl:", item["pos"].lower()),
                                              item["examples"]))
    return DefinitionEntity(label, results_marshalling)


def find_most_related_entity(definition_entity: DefinitionEntity, descriptions) -> (List[str], np.array):
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
    # I can't find a way to make this vectorized, but it should not be a concern.
    correlation_scores = [query.dot(embedding_results.T) for query in embedding_queries]
    correlation_scores = np.stack(correlation_scores)
    
    # Match the queries with the entities based on maximum cosine similarity
    preferred = np.argmax(correlation_scores, axis=1)
    result = [definition_entity.senses[idx].id for idx in preferred]
    scores = np.max(correlation_scores, axis=1)
    
    return result, scores
