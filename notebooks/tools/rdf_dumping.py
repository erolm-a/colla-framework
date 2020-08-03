import base64
from collections import defaultdict

import mmh3
import pandas as pd
from rdflib import Graph, Literal, Namespace, namespace, URIRef

from .dumps import wrap_open
from .providers import WikidataProvider

with wrap_open(("wikidata/grammatical_categories")) as fp:
    wikidata_grammatical_categories = pd.read_json(fp)


with wrap_open(("wikidata/pos_categories")) as fp:
    pos_categories = pd.read_json(fp)


namespaces = {
    "dct": "http://purl.org/dc/terms/",
    "ontolex": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "wikibase": "http://wikiba.se/ontology#",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "kgl": "http://kgl.ir.dcs.gla.ac.uk/entity/",
    "kgl-prop": "http://kgl.ir.dcs.gla.ac.uk/property/"
}

namespaces = dict((key, Namespace(val)) for (key, val) in namespaces.items())

for ns in dir(namespace):
    imported = getattr(namespace, ns)
    if isinstance(imported, Namespace) or isinstance(imported, namespace.ClosedNamespace):
        namespaces[ns.lower()] = imported

kgl = namespaces["kgl"]
kgl_prop = namespaces["kgl-prop"]
form_link = namespaces["ontolex"].lexicalForm
sense_link = namespaces["ontolex"].sense
form_label = namespaces["ontolex"].representation
rdfs_label = namespaces["rdfs"].label
pos_link = kgl_prop.pos
sameAs = namespaces["owl"].sameAs
definition = namespaces["skos"].definition
grammaticalFeature = kgl_prop.grammaticalFeature

def populate_categories(g: Graph):
    category_dict = {}
    
    for row in wikidata_grammatical_categories.iterrows():
        label = row[1]['entityLabel.value']
        wikidata_identifier = row[1]['entity.value']
        grammatical_form = hash(label, "grammatical_category")
        category_dict[label] = kgl[grammatical_form]
        g.add((kgl[grammatical_form], rdfs_label, Literal(label)))
        g.add((kgl[grammatical_form], sameAs, URIRef(wikidata_identifier)))

    # Wikidata is a horrible mess
    # Apparently some of the most beefy categories are not (in)direct subclasses
    # of "grammatical categories".
    extra_noun_categories = ["countable", "uncountable", "irregular",
                                  "usually uncountable", "unattested plural", "uncertain plural"]

    for noun_cat in extra_noun_categories:
        cat_id = kgl[hash(label, "grammatical_category")]
        category_dict[noun_cat] = cat_id
        g.add((cat_id, rdfs_label, Literal(noun_cat)))

    return category_dict

def hash(word, pos):
    return bytes.decode(base64.b32encode(mmh3.hash_bytes(word + pos))).rstrip("=").lower()

def is_in_graph(x):
    try:
        next(g.triples((x, None, None)))
        return True
    except StopIteration:
        return False

form_counter = defaultdict(int)
sense_counter = defaultdict(int)


def add_form(g: Graph, word_id: str, lexeme_id: URIRef, form_label: str):
    count = form_counter[word_id]
    form_id = kgl[f"{word_id}-F{count}"]
    form_counter[word_id] += 1
    g.add((lexeme_id, form_link, form_id))
    g.add((form_id, rdfs_label, Literal(form_label, lang="en")))
    g.add((form_id, form_label, Literal(form_label, lang="en")))
    return form_id

def add_sense(g: Graph, word_id: str, lexeme_id: URIRef, sense_definition: str):
    count = sense_counter[word_id]
    sense_id = kgl[f"{word_id}-S{count}"]
    sense_counter[word_id] += 1
    g.add((lexeme_id, sense_link, sense_id))
    g.add((sense_id, rdfs_label, Literal(sense_definition, lang="en")))
    g.add((sense_id, definition, Literal(sense_definition, lang="en")))
    return sense_id
    
def add_to_graph(g, category_dict, word, senses, pos, noun_forms, adj_forms, verb_forms):
    word_id = hash(word, pos)
    lexeme_id = kgl[word_id]
    if not is_in_graph(word_id):
        g.add((lexeme_id, pos_link, kgl[pos]))
        g.add((lexeme_id, namespaces['rdfs'].label, Literal(word, lang="en")))
        # g.add((lexeme_id, namespace['dct'].language, something_for_english_language))
        
    
    # Detect collision by just looking at the word label.
    # In theory we should also check that different pos may cause a collision
    # but it looks extremely unlikely
    else:
        label = g.label(word_id)
        if label != word:
            print(f"Detected collision between {label} and {word}")
            word_id = hash(word + "$42", pos)
            lexeme_id = kgl[word_id]
            g.add((lexeme_id, pos_link, kgl[pos]))
            g.add((lexeme_id, namespaces['rdfs'].label, Literal(word, lang="en")))
    
    for sense in senses:
        glosses = sense['glosses']
        if glosses:
            for gloss in glosses:
                sense = add_sense(g, word_id, lexeme_id, gloss)
            
    if noun_forms:
        if noun_forms.irregular:
            g.add((lexeme_id, grammaticalFeature, category_dict["irregular"]))
        
        # countable can be either no or yes or sometimes.
        if not noun_forms.countable == "no":
            g.add((lexeme_id, grammaticalFeature, category_dict["countable"]))
        if not noun_forms.countable == "yes":
            if noun_forms.always:
                g.add((lexeme_id, grammaticalFeature, category_dict["uncountable"]))
            else:
                g.add((lexeme_id, grammaticalFeature, category_dict["usually uncountable"]))
            
        if noun_forms.optional:
            g.add((lexeme_id, grammaticalFeature, category_dict[optional + " plural"]))

        singular = add_form(g, word_id, lexeme_id, word_id)
        g.add((singular, grammaticalFeature, category_dict['singular']))
        
        for plural in noun_forms.plurals:
            form_id = add_form(g, word_id, lexeme_id, plural)
            g.add((form_id, kgl_prop['grammaticalFeature'], category_dict['plural']))
    
    if verb_forms:
        pass