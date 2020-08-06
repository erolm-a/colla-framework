# Let Python do the Flyweight for us :)

from .providers import FusekiProvider
import spacy
import logging

nlp = spacy.load("en_core_web_md")
fuseki_provider = FusekiProvider()
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def replace_logger(_logger):
    global logger
    logger = _logger