"""
Wrapper for a grammar handler (in this case, SippyCup).

To use this grammar, simply use the exported `grammar` object.

Example:

```
>>> parses = grammar.parse_input('definition for jungle')
>>> for parse in parses: print(parse.semantics)
{intent: definition, np: jungle}
```

A parse may return multiple parses with slightly different semantics.
Use `pick_best_semantics` to fetch the "most likely" semantics (dangerous).
"""

import sys
sys.path.append('../../notebooks/3rdparty/sippycup')

import spacy
from annotator import *
from parsing import *
from typing import List
from functools import reduce

from .strings import convert_ordinal

from .globals import nlp, fuseki_provider, logger



def add_rule(grammar: Grammar, rule: Rule):
    """
    Refer to SippyCup's `add_rule` function.
    This function overrides SippyCup's `add_rule` by also handling
    optionals and mixed terminals and non-terminals in n-ary ryules"""
    if contains_optionals(rule):
        add_rule_containing_optional(grammar, rule)
    elif is_lexical(rule):
        grammar.lexical_rules[rule.rhs].append(rule)
    elif is_unary(rule):
        grammar.unary_rules[rule.rhs].append(rule)
    elif is_binary(rule):
        grammar.binary_rules[rule.rhs].append(rule)
    elif all([is_cat(rhsi) for rhsi in rule.rhs]):
        add_n_ary_rule(grammar, rule)
    else:
        make_cat(grammar, rule)


def add_rule_containing_optional(grammar: Grammar, rule: Rule):
    """
    See SippyCup's `add_rule_containing_optional` docstring.
    """
    # Find index of the first optional element on the RHS.
    first = next((idx for idx, elt in enumerate(rule.rhs) if is_optional(elt)), -1)
    assert first >= 0
    assert len(rule.rhs) > 1, 'Entire RHS is optional: %s' % rule
    prefix = rule.rhs[:first]
    suffix = rule.rhs[(first + 1):]
    # First variant: the first optional element gets deoptionalized.
    deoptionalized = (rule.rhs[first][1:],)
    add_rule(grammar, Rule(rule.lhs, prefix + deoptionalized + suffix, rule.sem))
    # Second variant: the first optional element gets removed.
    # If the semantics is a value, just keep it as is.
    sem = rule.sem
    # But if it's a function, we need to supply a dummy argument for the removed element.
    if isinstance(rule.sem, FunctionType):
        sem = lambda sems: rule.sem(sems[:first] + [None] + sems[first:])
    add_rule(grammar, Rule(rule.lhs, prefix + suffix, sem))


def make_cat(grammar: Grammar, rule: Rule):
    """
    Convert a terminal in the RHS into a non-terminal.
    
    Conversion works by creating a nonterminal from each terminal if
    it does not exist already in the grammar, otherwise it just replaces it.
    """
    
    new_rhs = []
    for rhsi in rule.rhs:
        if is_cat(rhsi):
            cat_name = rhsi
        else:
            cat_name = "$" + rhsi + "__nonterminal"
            if cat_name not in grammar.categories:
                grammar.categories.add(cat_name)
                # print(f"Adding rule: {cat_name} := {str(rhsi)}")
                add_rule(grammar, Rule(cat_name, rhsi))
        new_rhs.append(cat_name)
        # print(f"Adding rule: {rule.lhs} := {str(new_rhs)}")
    add_rule(grammar, Rule(rule.lhs, tuple(new_rhs), rule.sem))


def parse_input(grammar: Grammar, input: str, must_consume_all=True):
    """
    Returns a list of all parses for input using the given grammar.
    
    Note: the rules are case-sensitive and perfectly token-sensitive.
    
    params:

    - `grammar`: the Grammar instance to use
    - `input`: the lowercase'd string to use.
    - `must_consume_all`: if True the string must parse all the tokens.
      Otherwise return all the parses that match with the $ROOT symbol.

    """
    tokens_spacy = nlp(input) # New
    tokens = [token.text for token in tokens_spacy]
    # the chart of the CYK parsing algorithm.
    chart = defaultdict(list)
    for j in range(1, len(tokens) + 1):
        for i in range(j - 1, -1, -1):
            apply_annotators(grammar, chart, tokens, i, j)
            apply_lexical_rules(grammar, chart, tokens, i, j)
            apply_binary_rules(grammar, chart, i, j)
            apply_unary_rules(grammar, chart, i, j)

    all_parses = []
    for i in range(len(tokens), -1, -1):
        parses = chart[(0, i)]
    
        if hasattr(grammar, 'start_symbol') and grammar.start_symbol:
            all_parses.extend([parse for parse in parses if parse.rule.lhs == grammar.start_symbol])
        if len(parses) > 0 or must_consume_all:
            break

    return all_parses


class Grammar:
    """Redefinition of the Grammar class."""
    def __init__(self, rules=[], annotators=[], start_symbol='$ROOT'):
        self.categories = set()
        self.lexical_rules = defaultdict(list)
        self.unary_rules = defaultdict(list)
        self.binary_rules = defaultdict(list)
        self.annotators = annotators
        self.start_symbol = start_symbol
        for rule in rules:
            add_rule(self, rule)
        logger.info('Created grammar with %d rules.' % len(rules))

    def parse_input(self, input: str):
        """Returns a list of parses for the given input."""
        return parse_input(self, input)

# Annotators

class StopWordAnnotator(Annotator):
    """Use SpaCy to detect stop words on a per-token basis"""
    def annotate(self, tokens):
        if len(tokens) == 1:
            if nlp(tokens[0])[0].is_stop:
                return [('$StopWord', tokens[0])]
        return []

class ShowVerbAnnotator(Annotator):
    """A "ShowVerb" is a verb (single token or phrasal verb) that asks for
       information. The English language has a discrete number of such verbs
       with similar semantics.
       This class uses word2vec to find similarity in intent to categorize such
       verbs.
    """
    def __init__(self, threshold = 0.7):
        self.show_verbs = [("define", ""), ("tell", "me"), ("show", "me")]
        self.spacy_show_toks = nlp(" ".join([verb for verb, _ in self.show_verbs]))
        self.threshold = 0.7

    def annotate(self, tokens):
        if len(tokens) <= 2:
            spacy_tokens = nlp(" ".join(tokens))
            spacy_token = spacy_tokens[0]
            if spacy_token.pos_ != 'VERB':
                return []
            
            # If the verb matches in meaning and, in case it requires a
            # follow-up word, that this matches as well, then it's a match.
            for idx, (verb, acc) in enumerate(self.show_verbs):
                spacy_verb = self.spacy_show_toks[idx]
                if spacy_token.similarity(spacy_verb) >= self.threshold:
                    if verb == tokens[0] and acc != "" and (len(tokens) == 1 or tokens[1] != acc):
                        return []
                    return [('$ShowVerb', tokens)]
        return []

class TokenAnnotatorBuilder(Annotator):
    """
    This class serves a similar purpose to SippyCup's `TokenAnnotator`, but
    allows to exclude certain tokens from being annotated, like a blacklist.
    """

    def __init__(self, category_name, excluded=[]):
        """
        params:
        - category_name: the name of the annotation
        - excluded: the list of token lexemes to ignore.
        """
        Annotator.__init__(self)
        self.category_name = category_name
        self.excluded = excluded
    
    def annotate(self, tokens):
        if len(tokens) == 1:
            token = tokens[0]
            if token not in self.excluded:
                return [(self.category_name, token)]
        return []

class OrdinalNumberAnnotator(Annotator):
    """
    Annotate ordinal numbers.
    """
    def annotate(self, tokens):
        if len(tokens) > 1:
            return []
        value = convert_ordinal(tokens[0])
        if value:
            return [('$OrdinalNumber', value)]
        return []

class POSAnnotator(Annotator):
    """
    Annotate parts of speech.
    Used in the filter intent matcher to determine if the user mentioned a part
    of speech.
    """
    def annotate(self, tokens):
        candidate = " ".join(tokens)
        value = None
        if candidate == "noun":
            value = "noun"
        if candidate == "verb":
            value = "verb"
        if candidate == "adjective":
            value = "adj"
        if candidate == "adverb":
            value = "adv"
        if candidate == "pronoun":
            value = "pron"
        if value:
            return [('$POS', value)]
        return []


class GrammaticalFeatureAnnotator:
    """
    Annotate a grammatical feature.
    Used to determine if the user mentioned a grammatical feature ("singular",
    "third person indicative" etc.);
    """
    def __init__(self, categories_set):
        self.categories_set = categories_set

    def annotate(self, tokens):
        # TODO: this is a stub, please improve
        candidate = " ".join(tokens)
        if candidate in self.categories_set:
            return [("$GrammaticalFeature", candidate)]
        return []

categories_set = set(fuseki_provider.fetch_all_grammatical_categories().to_numpy())


# Semantics tool functions

def sems_0(sems):
    """Return the first semantics"""
    return sems[0]

def sems_1(sems):
    """Return the second semantics"""
    return sems[1]

def sems_2(sems):
    """Return the third semantics"""
    return sems[2]

def merge_dicts(d1, d2):
    """
    Merge the given dictionaries.
    Handles the nullity of either or both the dictionaries.
    """
    if not d2:
        return d1
    if not d1:
        return {}
    return {**d1, **d2}

def strip_none(sems):
    """Return the non-null semantics, if any"""
    return [sem for sem in sems if sem]

def merge_dicts_singleparam(sems):
    """merge the given semantics"""
    if all([sem is None for sem in sems]):
        return {}
    return reduce(merge_dicts, strip_none(sems))

def to_np(sems):
    """Wrap into a np key"""
    return {'np': strip_none(sems)[0]}

def concatenate(sems):
    """Concatenate the semantics as strings.
    For non-string semantics, use `merge_dicts_singleparam`
    """
    return " ".join(strip_none(sems))

rules_definition = [
    Rule('$ROOT', '$DefinitionQuery', sems_0),
    Rule('$DefinitionQuery', '$DefinitionQueryElements',
         lambda sems: merge_dicts({'intent': 'definition'}, sems[0])),
    Rule('$DefinitionQueryElements', '$DefinitionQuestion $NounPhrase',
         merge_dicts_singleparam),
    
    # Special case: "what does X mean?"
    Rule('$DefinitionQueryElements', 'what does $NounPhrase mean', sems_2),
    
    Rule('$DefinitionQuestion', '$ShowVerb ?me ?$Determiner'),
    Rule('$DefinitionQuestion', '$ShowVerb ?me $WhoDefinition'),
    Rule('$DefinitionQuestion', '$ShowVerb ?me $WhatDefinition'),
    Rule('$DefinitionQuestion', '$WhatDefinition'),
    Rule('$DefinitionQuestion', '$WhoDefinition', {'isPerson': True}),
    Rule('$WhoDefinition', 'who $Be'),
    Rule('$WhatDefinition', 'what $Be ?$Determiner ?$DefinitionFor'),
    Rule('$WhatDefinition', 'how do you $ShowVerb'),
    Rule('$DefinitionFor', '$WordSense $StopWord'),
    Rule('$NounPhrase', "$Tokens", to_np),
    Rule('$NounPhrase', "' $Tokens '", to_np),
    Rule('$NounPhrase', '" $Tokens "', to_np),
    Rule('$Tokens', '$UnquotedToken ?$Tokens', concatenate)
]

rules_determiner = [
    Rule('$Determiner', 'a'),
    Rule('$Determiner', 'an'),
    Rule('$Determiner', 'the'),
    Rule('$Determiner', 'about the'),
    Rule('$Determiner', 'its'),
]

rules_be = [
    Rule("$Be", "is"),
    Rule("$Be", "are"),
    Rule("$Be", "'s"),
    Rule("$Be", "were"),
    Rule("$Be", "was"),
]

rules_wordsenses = [
    Rule("$WordSense", "one"),
    Rule("$WordSense", "sense"),
    Rule("$WordSense", "meaning"),
    Rule("$WordSense", "definition"),
    Rule("$WordSense", "definitions"),
    Rule("$WordSense", "possibility"),
    Rule("$WordSense", "possibilities"),
    Rule("$WordSense", "case"),
    Rule("$WordSense", "field"),
]



def merge_dict_type_builder(type_):
    def f(sems):
        return merge_dicts({'type': type_, 'value': sems[4]}, sems[1])
    return f

rules_filter = [
    Rule('$ROOT', '$FilterQuery', lambda sems: merge_dicts({'intent': 'filter'}, sems[0])),
    # Tell me about...
    Rule('$FilterQuery', '?$ShowVerb ?$StopWord $FilterQueryElements', sems_2),
    # What about...
    Rule('$FilterQuery', 'what about $FilterQueryElements', sems_2),
    # What are the...
    Rule('$FilterQuery', 'what $Be ?$Determiner $FilterQueryElements', lambda sems: sems[3]),
    # "which examples are available?"
    Rule('$FilterQuery', 'what $FilterQueryElements $be $More', sems_1),
    Rule('$FilterQuery', 'which $FilterQueryElements $be $More', sems_1),
    Rule('$FilterQuery', '$FilterQueryElements', sems_0),
    
    
    # ordinal case
    Rule('$FilterQueryElements', "?$More the $OrdinalNumber ?$WordSense ?$Only",
         lambda sems: {'filtertype': 'number', 'value': strip_none(sems)[0]}),
         
    # "more about the mathematical case"
    Rule('$FilterQueryElements', "?$More the $UnquotedToken $WordSense ?$Only",
         lambda sems: {'filtertype': 'semantic', "value": strip_none(sems)[0]}),
    
    # some examples
    Rule('$FilterQueryElements', '?$More $Extra', sems_1),
    # some examples for the second case
    Rule('$FilterQueryElements', '?$More $Extra $StopWord ?$Determiner $OrdinalNumber $WordSense ?$Only',
         #lambda sems: merge_dicts({'type': 'number', 'value': sems[4]}, sems[1])),
         merge_dict_type_builder('number')),
         
    # some examples for the botanical case
    Rule('$FilterQueryElements', '?$More $Extra $StopWord ?$Determiner $UnquotedToken $WordSense ?$Only',
         # lambda sems: merge_dicts({'type': 'sense_meaning', 'value': sems[4]}, sems[1])),
         merge_dict_type_builder('semantic')),

    # some examples as a verb
    Rule('$FilterQueryElements', '?$More $Extra ?$Filler $StopWord ?$Determiner $POS',
         # lambda sems: merge_dicts({'type': 'sense_meaning', 'value': sems[4]}, sems[1])),
         lambda sems: merge_dicts({'filtertype': 'grammatical', 'requiredPos': sems[5]}, sems[1])),
    
    # Show me the plural form
    Rule("$FilterQueryElements", "$Determiner $GrammaticalFeature ?form",
         lambda sems: {'filtertype': 'grammatical', 'grammaticalFeature': sems[1]}),
    
    # Ask for examples, categories or usages
    Rule('$Extra', 'examples', {'variant': "example"}),
    Rule('$Extra', 'categories', {'variant': "categories"}),
    Rule('$Extra', 'usages', {'variant': "usages"}),
    Rule('$Extra', 'senses', {'variant': "senses"}),
    Rule('$Extra', 'parts of speech', {'variant': "pos"}),
    Rule('$Extra', 'conjugate', {'variant': "forms"}),
    Rule('$Extra', 'conjugation', {'variant': "forms"}),
    Rule('$Extra', 'forms', {'variant': "forms"}),
    
    # Category question where category precedes the rest
    Rule('$FilterQuery', "$FilterCategoryQuery",
         lambda sems: merge_dicts({'filtertype': 'semantic'}, sems[0])),
    # in the field of computer science, what does x mean?
    Rule('$FilterCategoryQuery', "$Category $WhatFilter", sems_0),
    Rule('$FilterCategoryQuery', "$WhatFilter $Category", sems_1),
    Rule('$FilterCategoryQuery', "$Category $?More $Extra", merge_dicts_singleparam),
    
    
    Rule('$More', "more"),
    Rule('$More', "more about"),
    Rule('$More', "some"),
    Rule('$More', "some some"),
    Rule('$More', 'possible'),
    Rule('$More', 'available'),
    

    
    Rule("$Only", "only"),
    Rule("$Only", "alone"),
    
    Rule("$Filler", "$StopWord $NounPhrase"),
    
    Rule("$Category", "in $Determiner $WordSense $StopWord $NounPhrase ?,", lambda sems: {'category': sems[4]['np']}),
    Rule("$WhatFilter", "what does $NounPhrase mean"),
    Rule("$WhatFilter", "what $Be $Determiner $WordSense"),
]


rules_derived = [
    Rule('$ROOT', '$RelatedQuery', lambda sems: merge_dicts({'intent': 'related'}, sems[0])),
    
    Rule('$RelatedQuery', '?$ShowVerb $RelatedQueryElements', sems_1),
    # What are related senses?
    Rule('$RelatedQuery', 'what $Be $RelatedQueryElements', sems_2),
    Rule('$RelatedQuery', 'which $Be $RelatedQueryElements', sems_2),
    # What senses are related
    Rule('$RelatedQuery', 'what $Word $Be $RelatedQueryElements', lambda sems: sems[3]),
    Rule('$RelatedQuery', '$RelatedQueryElements', sems_0),
    Rule('$RelatedQueryElements', '?$Determiner $Derived ?$Word', sems_1),
    Rule('$RelatedQueryElements', '?$More $Derived ?$Word', sems_1),
    Rule('$RelatedQueryElements', '?$Determiner $Quality $Derived', lambda sems: merge_dicts(sems[1], sems[2])),
    Rule('$RelatedQueryElements', '?$More $Quality $Derived', lambda sems: merge_dicts(sems[1], sems[2])),
    
    
     # some examples of the derived words
    Rule('$RelatedQueryElements', '?$More $Extra $StopWord ?$Determiner $Derived ?$Word',
         lambda sems: merge_dicts(sems[1], sems[4])),
         
    Rule('$Derived', 'derived', {'filtertype': 'derived'}),
    Rule('$Derived', 'synonym', {'filtertype': 'synonym'}),
    Rule('$Derived', 'synonyms', {'filtertype': 'synonym'}),
    Rule('$Derived', 'antonym', {'filtertype': 'antonym'}),
    Rule('$Derived', 'opposites', {'filtertype': 'antonym'}),
    Rule('$Derived', 'antonyms', {'filtertype': 'antonym'}),
    
    Rule('$Quality', '$QualityToken', lambda sems: {'category': sems[0]}),
]

rules_words = [
    Rule('$Word', 'word'),
    Rule('$Word', 'words'),
    Rule('$Word', 'lexeme'),
    Rule('$Word', 'lexemes'),
    Rule('$Word', 'lemma'),
    Rule('$Word', 'lemmas'),
]


ruleset = rules_be + rules_definition + rules_determiner + \
          rules_filter + rules_words + rules_wordsenses + \
          rules_derived
annotators = [StopWordAnnotator(), ShowVerbAnnotator(),
                TokenAnnotatorBuilder("$UnquotedToken", ["'", '"', "?", ","]), # commas must split noun phrases
                OrdinalNumberAnnotator(), POSAnnotator(),
                GrammaticalFeatureAnnotator(categories_set)]

# Entry point of the grammar
grammar = Grammar(rules=ruleset, annotators=annotators)

def pick_best_semantics(parses):
    """
    Return the most likely matching parse.
    
    This is a simple stub. Does not do any ML here, despite it could
    (and should), so use with care.
    """
    if parses == []:
        return {}
    semantics = [parse.semantics for parse in parses]
    
    if all(parse["intent"] == "definition" for parse in semantics):
        picked_parser = min(semantics, key=lambda parse: len(parse["np"]))
    
    else:
        priority = {'grammatical': 1, 'semantic': 2, 'number': 3}
        picked_parser = max(semantics, key=lambda parse: len(parse.keys()) * 10 + (priority[parse['filtertype']] if 'filtertype' in parse else 0))
        
    return picked_parser

def print_parse(utterances):
    """
    Print all the parses for the given utterances.
    This function is provided for inspection.

    `utterances` is a list of strings to parse.
    """
    for utterance in utterances:
        logger.debug("=" * 20)
        logger.debug("For the utterance " + utterance + ":")
        for parse in grammar.parse_input(utterance):
            logger.debug(parse.semantics)