'''
Some tools to cache datasets and query them

WARNING: It has been written before we moved to Wiktionary and BabelNet.
This module has been kept for historical relevance but should be removed.
'''

import os
import json
import requests
import pandas as pd

from .sparql_wrapper import dbpedia_sparql, wikidata_sparql
from .strings import strip_prefix
from .dumps import *
from .providers import WikidataProvider, DBPediaProvider

def fetch_dataset(entity:str, provider:str, flavor : str = None, force_redownload: bool = False, format: str = "json", *args, **kwargs):
    """Fetch a dataset. This function abstracts out the mechanics of
    dataset harvesting and dumping.
    
    Accepted values for provider:
    
    - "wikidata"
    - "dbpedia"
    
    Accepted values for flavor (supported by Wikidata provider only):
    
    - None (default): extract all data (including metadata and qualifiers)
    - simple: extract all tuples minus metadata and qualifiers
    """
    
    base_provider = None
    if provider == "wikidata":
        base_provider = WikidataProvider()
    elif provider == "dbpedia":
        base_provider = DBPediaProvider()
    else:
        raise Exception(f"Unsupported provider: {provider}")
        
    base_provider.fetch_dataset(entity, flavor=flavor, format=format, force_redownload=force_redownload, *args, **kwargs)
    with wrap_open(base_provider.get_filename_path(entity, format)) as fp:
        return json.load(fp)


def get_wikidata_edges(entity_df, entities_only=True):
    """Generate triples: [<subject>, <predicate>, <object>].
        Only consider objects for which there is a linked entity
        
        If entities_only is True, consider the triples where the objects are wikidata entities.
    """
    
    root = entity_df["entities"]
    subjects = list(entity_df["entities"].keys())
    
    predicates = [(subject, claim, root[subject]["claims"][claim]) for subject in subjects for claim in root[subject]["claims"]]

    tuples = []
    
    for (subject, claim, snaks) in predicates:
        for snak in snaks:
            mainsnak = snak["mainsnak"]
            if mainsnak["snaktype"] == "value":
                datavalue = mainsnak["datavalue"]
                datavalue_value = datavalue["value"]
                datavalue_type = datavalue["type"]     
                
                if datavalue_type == "wikibase-entityid":
                    obj_to_append = datavalue_value["id"]
                elif entities_only:
                    continue
                elif datavalue_type == "string":
                    obj_to_append = datavalue_value
                else:
                    obj_to_append = None
                
                if obj_to_append:
                    tuples.append((subject, claim, obj_to_append))
            else:
                print("snaktype != value", mainsnak["snaktype"])
            
    
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