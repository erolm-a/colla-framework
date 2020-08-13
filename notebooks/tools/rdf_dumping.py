import base64
from collections import defaultdict

from enum import Enum

import mmh3
import pandas as pd
from rdflib import Graph, Literal, Namespace, namespace, URIRef
from rdflib.extras.infixowl import Class

from .dumps import wrap_open
from .providers import WikidataProvider

with wrap_open("wikidata/grammatical_categories.json") as fp:
    wikidata_grammatical_categories = pd.read_json(fp)

with wrap_open("wikidata/pos_categories.json") as fp:
    pos_categories = pd.read_json(fp)


# Namespaces we are going to use here...
namespaces = {
    "dct": "http://purl.org/dc/terms/",
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "wikibase": "http://wikiba.se/ontology#",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "kgl": "http://grill-lab.org/kg/entity/",
    "kglprop": "http://grill-lab.org/kg/property/"
}

namespaces = dict((key, Namespace(val)) for (key, val) in namespaces.items())

# Import default namespaces from the library
for ns in dir(namespace):
    imported = getattr(namespace, ns)
    if isinstance(imported, Namespace) or isinstance(imported, namespace.ClosedNamespace):
        namespaces[ns.lower()] = imported

kgl = namespaces["kgl"]
kgl_prop = namespaces["kglprop"]
form_link = namespaces["ontolex"].lexicalForm
kgl_form_link = kgl_prop.form
sense_link = namespaces["ontolex"].sense
kgl_sense_link = kgl_prop.sense
form_label = namespaces["ontolex"].representation
rdfs_label = namespaces["rdfs"].label
rdf_type = namespaces["rdf"].type
pos_link = kgl_prop.pos
sameAs = namespaces["owl"].sameAs
definition = namespaces["skos"].definition
kgl_definition = kgl_prop.definition
grammaticalFeature = kgl_prop.grammaticalFeature
kgl_label = kgl_prop.label
example_link = kgl_prop.example


class ExtractorGraph:
    """A Wrapper to rdflib's Graph that extracts from Wiktionary and BabelNet."""
    class SenseType(Enum):
        SENSE = 1,
        SUBSENSE = 2,
        USAGE = 3
    
    def __init__(self):
        g = Graph()
        self.g = g

        # Populate namespaces
        for k,v in namespaces.items():
            g.bind(k, v)

        lexemeClass = Class(kgl.Lexeme,
                                nameAnnotation=Literal("Lexeme"),
                                graph=g)
        lexemeClass.comment = Literal("A lexeme is the main entry of the dictionary")

        formClass = Class(kgl.Form, nameAnnotation=Literal("Form"), graph=self.g)
        formClass.comment = Literal("A form is a morphological form that appears when the lexeme is a declinable or conjugable noun")

        senseClass = Class(kgl.Sense, nameAnnotation=Literal("Sense"), graph=self.g)
        senseClass.comment = Literal("A sense, or synset, is a unit of meaning of a lexeme")

        subsenseClass = Class(kgl.Subsense, nameAnnotation=Literal("Subsense"), graph=self.g)
        subsenseClass.comment = Literal("A subsense is a possible refinement on a sense")

        usageClass = Class(kgl.Usage, nameAnnotation=Literal("Usage"), graph=self.g)
        usageClass.comment = Literal("An usage is a linguistic 'usage' of a lexeme")
        
        self.form_counter = defaultdict(int)
        self.sense_counter = defaultdict(int)

        grammaticalCategory = Class(kgl.GrammaticalCategory, graph=g)
        
        self.populate_categories()

    @staticmethod
    def hash(word, pos):
        """
        Tool function to generate unique identifiers for the lexemes.
        """
        mmhash = mmh3.hash64(word + pos, signed=False)[0]
        mmhash = int.to_bytes(mmhash, 8, "big")
        return bytes.decode(base64.b32encode(mmhash)).rstrip("=").lower()
    
    def is_in_graph(self, x):
        """
        Tool function to check if an entity is in the graph.
        """
        try:
            next(self.g.triples((x, None, None)))
            return True
        except StopIteration:
            return False

    def populate_categories(self):
        """
        Init the graph by harvesting a list of categories.
        """
        category_dict = {}
        g = self.g
        self.categories = category_dict

        for row in wikidata_grammatical_categories.iterrows():
            label = row[1]['entityLabel.value']
            wikidata_identifier = row[1]['entity.value']
            grammatical_form = self.hash(label, "grammatical_category")
            category_dict[label] = kgl[grammatical_form]
            g.add((kgl[grammatical_form], rdfs_label, Literal(label)))
            g.add((kgl[grammatical_form], sameAs, URIRef(wikidata_identifier)))

        # Wikidata is a horrible mess
        # Apparently some of the most beefy categories are not (in)direct subclasses
        # of "grammatical categories".
        extra_noun_categories = ["countable", "uncountable", "irregular",
                                      "usually uncountable", "unattested plural",
                                      "uncertain plural"]

        extra_verb_categories = ["defective"]

        extra_adjective_categories = ["positive", "comparative", "superlative",
                                            "not comparable", "comparable-only",
                                            "generally not comparable"]

        for cat in extra_noun_categories + extra_verb_categories + \
                     extra_adjective_categories:
                cat_id = self.add_category(cat)
                category_dict[cat] = cat_id
        
    def add_category(self, label):
        g = self.g
        cat_id = kgl[self.hash(label, "grammatical_category")]
        g.add((cat_id, rdfs_label, Literal(label)))
        g.add((cat_id, kgl_label, Literal(label)))
        g.add((cat_id, rdf_type, kgl.GrammaticalCategory))
        return cat_id

    def add_form(self, word_id: str, lexeme_id: URIRef, label: str):
        """
        Add a form.
        """
        g = self.g
        count = self.form_counter[word_id]
        form_id = kgl[f"{word_id}-F{count}"]
        self.form_counter[word_id] += 1
        g.add((lexeme_id, form_link, form_id))
        g.add((lexeme_id, kgl_form_link, form_id))
        g.add((form_id, namespaces['rdf'].type, kgl.Form))
        g.add((form_id, kgl_prop['label'], Literal(label, lang="en")))
        g.add((form_id, rdfs_label, Literal(label, lang="en")))
        g.add((form_id, form_label, Literal(label, lang="en")))
        return form_id


    def add_sense(self, word_id: str, lexeme_id: URIRef,
                     sense_definition: str, parent_sense_id=None,
                     sense_type=SenseType.SENSE):
        """
        Add a single sense/subsense/usage.
        """
        g = self.g
        count = self.sense_counter[word_id]
        sense_id = kgl[f"{word_id}-S{count}"]
        self.sense_counter[word_id] += 1

        if sense_type == self.SenseType.SENSE:
            g.add((lexeme_id, sense_link, sense_id))
            g.add((lexeme_id, kgl_sense_link, sense_id))
            g.add((sense_id, namespaces['rdf'].type, kgl.Sense))
        elif sense_type == self.SenseType.SUBSENSE:
            g.add((parent_sense_id, kgl_prop['subsense'], sense_id))
            g.add((sense_id, namespaces['rdf'].type, kgl.Subsense))
        else:
            g.add((parent_sense_id, kgl_prop['usage'], sense_id))
            g.add((sense_id, namespaces['rdf'].type, kgl.Usage))


        g.add((sense_id, rdfs_label, Literal(sense_definition, lang="en")))
        g.add((sense_id, definition, Literal(sense_definition, lang="en")))
        g.add((sense_id, kgl_definition, Literal(sense_definition, lang="en")))

        return sense_id

    def add_sense_rec(self, senses, word_id, lexeme_id, depth=0, parent_sense=None):
        """
        Recursively add a sense/subsense/usage to a lexeme/sense/subsense.
        """
        g = self.g
        for sense in senses:
            # TODO: now that senses are hierarchically structured, glosses should become a single string
            gloss = sense['glosses'][0] if sense['glosses'] else ""
            examples = sense['examples']
            if gloss:
                senseType = {0: self.SenseType.SENSE,
                               1: self.SenseType.SUBSENSE,
                               2: self.SenseType.USAGE}
                sense_id = self.add_sense(word_id, lexeme_id, gloss, parent_sense, senseType[depth])

                if examples:
                    for example in examples:
                        if example:
                            g.add((sense_id, example_link, Literal(example, lang="en")))

            if 'subsenses' in sense and sense['subsenses'] is not None:
                self.add_sense_rec(sense['subsenses'], word_id, lexeme_id, depth+1, sense_id)
            if 'usages' in sense and sense['usages'] is not None:
                self.add_sense_rec(sense['usages'], word_id, lexeme_id, depth+1, sense_id)

    def add_grammatical_categories(self, word_id, cats):
        """
        Add a list of grammatical categories to a given word
        """
        g = self.g
        for cat in cats:
            g.add((word_id, grammaticalFeature, self.categories[cat]))

    def add_noun_forms(self, word, word_id, lexeme_id, noun_forms):
        """
        Add noun forms
        """
        g = self.g
        if noun_forms.irregular:
            g.add((lexeme_id, grammaticalFeature, self.categories["irregular"]))

        # countable can be either no or yes or sometimes.
        if not noun_forms.countable == "no":
            g.add((lexeme_id, grammaticalFeature, self.categories["countable"]))
        if not noun_forms.countable == "yes":
            if noun_forms.always:
                g.add((lexeme_id, grammaticalFeature, self.categories["uncountable"]))
            else:
                g.add((lexeme_id, grammaticalFeature, self.categories["usually uncountable"]))

        if noun_forms["optional"]:
            self.add_grammatical_categories(lexeme_id,
                                                  [noun_forms["optional"] + " plural"])

        singular = self.add_form(word_id, lexeme_id, word)
        g.add((singular, grammaticalFeature, self.categories['singular']))

        if noun_forms["plurals"]:
            for plural in noun_forms.plurals:
                form_id = self.add_form(word_id, lexeme_id, plural)
                self.add_grammatical_categories(form_id, ['plural'])


    def add_adj_forms(self, word,  word_id, lexeme_id, adj_forms):
        """
        Add adjective forms
        """
        opt = adj_forms['optional']
        if adj_forms['optional']:
            self.add_grammatical_categories(lexeme_id, [opt])
        if opt is None or opt != "not comparable":
            positive_form = self.add_form(word_id, lexeme_id, word)
            self.add_grammatical_categories(positive_form, ["positive"])

        if adj_forms['comparatives']:
            for comp in adj_forms['comparatives']:
                comp_form = self.add_form(word_id, lexeme_id, comp)
                self.add_grammatical_categories(comp_form, ["comparative"])

        if adj_forms['superlatives']:
            for sup in adj_forms['superlatives']:
                sup_form = self.add_form(word_id, lexeme_id, sup)
                self.add_grammatical_categories(sup_form, ["superlative"])


    def add_verb_forms(self, word: str, word_id, lexeme_id, verb_forms):
        infinitive = self.add_form(word_id, lexeme_id, word)
        self.add_grammatical_categories(infinitive,
                                        ["present tense", "infinitive",
                                         "first-person singular",
                                         "second-person singular",
                                         "first-person plural",
                                         "second-person plural", 
                                         "third-person plural"])

        if verb_forms["pres_3sg"]:
            pres_3sg = self.add_form(word_id, lexeme_id, verb_forms["pres_3sg"])
            self.add_grammatical_categories(pres_3sg,
                                       ["present tense",
                                        "third-person singular"])

        else:
            self.add_grammatical_categories(lexeme_id, ["defective"])

        pres_ptc = self.add_form(word_id, lexeme_id, verb_forms["pres_ptc"])
        self.add_grammatical_categories(pres_ptc, ["present participle"])

        past = self.add_form(word_id, lexeme_id, verb_forms["past"])
        self.add_grammatical_categories(past, ["past tense", "simple past"])

        past_ptc = self.add_form(word_id, lexeme_id, verb_forms["past_ptc"])
        self.add_grammatical_categories(past_ptc, ["past participle"])

    def add_wiktionary(self, row):
        """
        Add a Dataframe row about a Wiktionary word to the graph.
        """
        g = self.g
        word = row['word']
        senses = row['senses']
        pos = row['pos']
        noun_forms = row['noun_forms']
        adj_forms = row['adj_forms']
        verb_forms = row['verb_forms']

        word_id = self.hash(word, pos)
        lexeme_id = kgl[word_id]
        if not self.is_in_graph(word_id):
            g.add((lexeme_id, namespaces['rdf'].type, kgl.Lexeme))
            g.add((lexeme_id, pos_link, kgl[pos]))
            g.add((lexeme_id, kgl_label, Literal(word, lang="en")))
            g.add((lexeme_id, rdfs_label, Literal(word, lang="en")))
            # g.add((lexeme_id, namespace['dct'].language, something_for_english_language))


        # Detect collision by just looking at the word label.
        # In theory we should also check that different pos may cause a collision
        # but it looks extremely unlikely
        # Unless the word has doubled entries. On that case, we ignore the problem (TODO)
        else:
            label = g.label(word_id)
            if label != word:
                print(f"Detected collision between {label} and {word}")
                word_id = self.hash(word + "$42", pos)
                lexeme_id = kgl[word_id]
                g.add((lexeme_id, pos_link, kgl[pos]))
                g.add((lexeme_id, kgl_prop.label, Literal(word, lang="en")))
                g.add((lexeme_id, namespaces['rdfs'].label, Literal(word, lang="en")))

        if row['senses']:
            self.add_sense_rec(row['senses'], word_id, lexeme_id)

        # Nouns
        if noun_forms:
            self.add_noun_forms(word, word_id, lexeme_id, noun_forms)

        # Adjectives
        if adj_forms:
            self.add_adj_forms(word, word_id, lexeme_id, adj_forms)

        # Verbs
        if verb_forms:
            self.add_verb_forms(word, word_id, lexeme_id, verb_forms)