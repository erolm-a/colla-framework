'''
Some tools to cache datasets and query them
'''

import os
import json
import requests
import pandas as pd

from .sparql_wrapper import dbpedia_sparql, wikidata_sparql
from .strings import strip_prefix

DATA_FOLDER = "data"


def get_filename_path(filename):
    return os.path.join(DATA_FOLDER, filename)


def wrap_open(filename, *args, **kwargs):
    """wrapper to python's open() that prepends DATA_FOLDER"""
    path = get_filename_path(filename)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    if "handler" in args:
        handler = args["handler"]
    else:
        handler = open
    return handler(path, *args, **kwargs)


def download_to(url: str, filename: str):
    r = requests.get(url)
    with wrap_open(filename, "wb") as f:
        f.write(r.content)

        
def fetch_wikidata_dataset(entity, _format="json", flavor=None):
    url = get_wikidata_link(entity, _format=_format, flavor=flavor)
    filename = entity + "." + _format
    
    if os.path.isfile(get_filename_path(filename)):
        print(f"Dataset {filename} already downloaded. Skipping...")
    else:
        print(f"Dataset {filename} not available, downloading from {url}")
        r = requests.get(url)
        with wrap_open(filename, "wb") as f:
            f.write(r.content)
        print("Done")


def get_wikidata_link(entity, _format="json", flavor=None) -> str:
    """
    Extract a wikidata entity link. Very simple and crude. More sofisticated
    queries should use a SPARQLWrapper.
    """
    
    url = f"https://wikidata.org/wiki/Special:EntityData/{entity}.{_format}"
    if flavor is not None:
        url += f"?flavor={flavor}"
    return url

def get_dbpedia_link(entity, _format="json") -> str:
    """Extract a dbpedia entity. Very simple and crude. More sofisticated
    queries should use a SPARQLWrapper.
    
    Note that in DBPedia wikipedia links are entity identifiers and are
    case-sensitive. The URL fetcher seems *not* to be solving the redirects
    alone.
    """
    return f"https://dbpedia.org/data/{entity}.{_format}"
    

def fetch_dbpedia_dataset(name: str, _format="json", force_redownload=False):
    """Download and cache a dbpedia dataset.
    
    Note that in DBPedia wikipedia links are entity identifiers and are
    case-sensitive. The URL fetcher seems *not* to be solving the redirects alone,
    but it is not guaranteed to always be the case. For such reasons, please
    avoid using a SPARQL's "DESCRIBE" query to dump an entity."""
    filename = name + "." + _format
    url = get_dbpedia_link(name, _format)
    
    if os.path.isfile(get_filename_path(name) and not force_redownload):
        print(f"Dataset {filename} already downloaded. Skipping...")
    else:
        print(f"Dataset {filename} already available. Downloading from {url}")
        download_to(url, filename)
        print("Done")
        

def fetch_dataset(entity:str, provider:str, flavor : str = None, force_redownload : bool = False):
    """Fetch a dataset. This function abstracts out the mechanics of
    dataset harvesting and dumping.
    
    Accepted values for provider:
    
    - "wikidata"
    - "dbpedia"
    
    Accepted values for flavor (supported by Wikidata provider only):
    
    - None (default): extract all data (including metadata and qualifiers)
    - simple: extract all tuples minus metadata and qualifiers
    """
    
    if provider == "wikidata":
        fetch_wikidata_dataset(entity, flavor=flavor)
        # TODO: possibly support other serialization options?
        with wrap_open(entity + ".json") as fp:
            return json.load(fp)
    
    elif provider == "dbpedia":
        fetch_dbpedia_dataset(entity, force_redownload=force_redownload)
        
        with wrap_open(entity + ".json") as fp:
            return json.load(fp)

    else:
        raise Exception(f"Unsupported provider: {provider}")


def wikidata_get_edges(entity_df):
    """Generate triples: [<subject>, <predicate>, <object>].
        Only consider objects for which there is a linked entity"""
    
    root = entity_df["entities"]
    subjects = list(entity_df["entities"].keys())
    
    predicates = [(subject, claim, root[subject]["claims"][claim]) for subject in subjects for claim in root[subject]["claims"]]

    tuples = []
    
    for (subject, claim, snaks) in predicates:
        for snak in snaks:
            mainsnak = snak["mainsnak"]
            if mainsnak["snaktype"] == "value" and mainsnak["datavalue"]["type"] == "wikibase-entityid":
                tuples.append((subject, claim, mainsnak["datavalue"]["value"]["id"]))
            
    
    return tuples

def sparql_values_in(l, prefix="wd:"):
    return " ".join([prefix + entity for entity in l])


def annotate_wikidata_entity(entity_list):
    query_list = sparql_values_in(entity_list)
    
    print(query_list)
    
    query = """
    SELECT ?entity ?label
    WHERE
    {
        VALUES ?entity {query_list}.
        ?entity rdfs:label ?label.
        FILTER(LANG(?label) = "en").
    }"""
    
    query = query.replace("query_list", query_list)
    return wikidata_sparql.run_query(query)
    

def annotate_wikidata_property(property_list):
    """Fetch the property name of a list of properties"""
    
    query_list = sparql_values_in(property_list)
    
    query = """
    SELECT ?property ?propertyLabel WHERE {
        ?property a wikibase:Property .
        VALUES ?property {?property_list}
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en" .
        }
    }
    """
    
    query = query.replace("?property_list", query_list)
    return wikidata_sparql.run_query(query)


def get_wikidata_edges(entity_df: dict):
    """Generate triples: [<subject>, <predicate>, <object>].
        Only consider objects for which there is a linked entity"""
    
    root = entity_df["entities"]
    subjects = list(entity_df["entities"].keys())
    
    predicates = [(subject, claim, root[subject]["claims"][claim]) for subject in subjects for claim in root[subject]["claims"]]

    tuples = []
    
    for (subject, claim, snaks) in predicates:
        for snak in snaks:
            mainsnak = snak["mainsnak"]
            if mainsnak["snaktype"] == "value" and \
               mainsnak["datavalue"]["type"] == "wikibase-entityid":
                tuples.append((subject, claim, mainsnak["datavalue"]["value"]["id"]))
            
    
    return tuples