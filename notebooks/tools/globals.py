# Let Python do the Flyweight for us :)

from .providers import FusekiProvider
import spacy

nlp = spacy.load("en_core_web_md")
fuseki_provider = FusekiProvider()