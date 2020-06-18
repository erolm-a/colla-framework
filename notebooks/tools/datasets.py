'''
Some tools to cache datasets and query them
'''

import os
import requests

from SPARQLWrapper import SPARQLWrapper, JSON

DATA_FOLDER = "data"
wikidata_sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
wikidata_sparql.setReturnFormat(JSON)


def get_filename_path(filename):
    return os.path.join(DATA_FOLDER, filename)


def wrap_open(filename, *args, **kwargs):
    """wrapper to python's open() that prepends DATA_FOLDER"""
    path = get_filename_path(filename)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    return open(path, *args, **kwargs)


def fetch_dataset(url, filename):
    if os.path.isfile(get_filename_path(filename)):
        print(f"File {filename} was already downloaded. Skipping...")
    else:
        print(f"Dataset not available, downloading from {url}")
        r = requests.get(url)
        with wrap_open(filename, "wb") as f:
            f.write(r.content)
        print("Done")


def get_wikidata_link(entity, _format="json", flavor=None):
    """
    Extract a wikidata entity. Very simple and crude. More sofisticated queries
    should use a SPARQLWrapper.
    """
    url = f"https://wikidata.org/wiki/Special:EntityData/{entity}.{_format}"
    if flavor is not None:
        url += f"?flavor={flavor}"
    return url


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


def annotate_entity(entity_list):
    query_list = sparql_values_in(entity_list)
    
    print(query_list)
    
    # multi-line f-strings seem to be broken with this version of jupyter.
    query = """
    SELECT ?entity ?label
    WHERE
    {
        VALUES ?entity {query_list}.
        ?entity rdfs:label ?label.
        FILTER(LANG(?label) = "en").
    }"""
    
    query = query.replace("query_list", query_list)
    wikidata_sparql.setQuery(query)
    results =  wikidata_sparql.query().convert()
    return pd.json_normalize(results['results']['bindings'])
    

def annotate_property(property_list):
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
    wikidata_sparql.setQuery(query)
    results =  wikidata_sparql.query().convert()
    return pd.json_normalize(results['results']['bindings'])