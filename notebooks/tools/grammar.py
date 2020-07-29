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


# TODO spacy is also being used somewhere else... make a flyweight?
nlp = spacy.load("en_core_web_md")

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
        print('Created grammar with %d rules.' % len(rules))

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

# Semantics tool functions

def sems_0(sems):
    """Return the first semantics"""
    return sems[0]

def sems_1(sems):
    """Return the second semantics"""
    return sems[1]

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
    
    Rule('$DefinitionQuestion', '$ShowVerb ?me'),
    Rule('$DefinitionQuestion', '$WhatDefinition'),
    Rule('$WhatDefinition', 'what is ?$Determiner ?$DefinitionFor'),
    Rule('$WhatDefinition', 'how do you $ShowVerb'),
    Rule('$DefinitionFor', 'meaning $StopWord'),
    Rule('$DefinitionFor', 'sense $StopWord'),
    Rule('$DefinitionFor', 'definition $StopWord'),
    Rule('$NounPhrase', "$Tokens", to_np),
    Rule('$NounPhrase', "' $Tokens '", to_np),
    Rule('$NounPhrase', '" $Tokens "', to_np),
    Rule('$Tokens', '$UnquotedToken ?$Tokens', concatenate)
]

def merge_dict_type_builder(type_):
    def f(sems):
        return merge_dicts({'type': type_, 'value': sems[4]}, sems[1])
    return f

rules_filter = [
    Rule('$ROOT', '$FilterQuery', sems_0),
    Rule('$FilterQuery', '?$ShowVerb $FilterQueryElements',
         lambda sems: merge_dicts({'intent': 'filter'}, sems[1])),
    
    Rule('$FilterQuery', 'what about $FilterQueryElements',
         lambda sems: merge_dicts({'intent': 'filter'}, sems[2])),
    
    # ordinal case
    Rule('$FilterQueryElements', "?$More the $OrdinalNumber ?$WordSense ?$Only",
         lambda sems: {'type': 'number', 'value': strip_none(sems)[0]}),
    
    # "more about the mathematical case"
    Rule('$FilterQueryElements', "?$More the $UnquotedToken $WordSense ?$Only",
         lambda sems: {'type': 'sense_meaning', "value": strip_none(sems)[0]}),
    
    # some examples
    Rule('$FilterQueryElements', '?$More $Extra', sems_1),
    # some examples for the second case
    Rule('$FilterQueryElements', '?$More $Extra $StopWord ?$Determiner $OrdinalNumber $WordSense ?$Only',
         merge_dict_type_builder('number')),
    
    # some examples for the botanical case
    Rule('$FilterQueryElements', '?$More $Extra $StopWord ?$Determiner $UnquotedToken $WordSense ?$Only',
         merge_dict_type_builder('sense_meaning')),
    
    
    Rule('$Extra', 'examples', {'variant': "example"}),
    Rule('$Extra', 'related words', {'variant': 'related'}),
    
    Rule('$More', "more"),
    Rule('$More', "more about"), # TODO: add optionals for terminals as well
    Rule('$More', "some"),
    
    Rule("$WordSense", "one"),
    Rule("$WordSense", "sense"),
    Rule("$WordSense", "meaning"),
    Rule("$WordSense", "definition"),
    Rule("$WordSense", "possibility"),
    Rule("$WordSense", "case"),
    
    Rule("$Only", "only"),
    Rule("$Only", "alone"),
]

rules_determiner = [
    Rule('$Determiner', 'a'),
    Rule('$Determiner', 'an'),
    Rule('$Determiner', 'the'),
]

ruleset = rules_definition + rules_determiner + rules_filter
annotators = [StopWordAnnotator(), ShowVerbAnnotator(),
                TokenAnnotatorBuilder("$UnquotedToken", ["'", '"', "?"]),
                OrdinalNumberAnnotator()]

# Entry point of the grammar
grammar = Grammar(rules=ruleset, annotators=annotators)

def pick_best_semantics(parses):
    """
    Return the most likely matching parse.
    
    This is a simple stub. Does not do any ML here, despite it could
    (and should), so use with care.
    """
    semantics = [parse.semantics for parse in parses]
    
    if all(parse["intent"] == "definition" for parse in semantics):
        picked_parser = min(semantics, key=lambda parse: len(parse["np"]))
    
    else:
        priority = {'sense_meaning': 1, 'number': 2}
        for sem in semantics:
            print(sem)
        picked_parser = max(semantics, key=lambda parse: len(parse.keys()) * 10 + (priority[parse['type']] if 'type' in parse else 0))
        
    return picked_parser
