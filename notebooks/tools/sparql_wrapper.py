"""Some data sources (DBPedia, Wikidata) provide a SPARQL endpoint. However, their ontologies differ radically.

Here we provide some tools to simplify queries by providing proper prefixes."""

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, JSONLD

from .strings import strip_prefix, remove_suffix

from functools import reduce

class SPARQLDataProviders():
    prefixes = {
        "owl": "http://www.w3.org/2002/07/owl#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "foaf": "http://xmlns.com/foaf/0.1/",
        "dc": "http://purl.org/dc/elements/1.1/",
        "": "http://dbpedia.org/resource/",
        "dbpedia2": "http://dbpedia.org/property/",
        "dbpedia": "http://dbpedia.org/",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "ace": "http://attempto.ifi.uzh.ch/ace_lexicon#",
        "kgl": "http://grill-lab.org/kg/entity/",
        "kglprop": "http://grill-lab.org/kg/property/",
        "wd": "http://www.wikidata.org/entity/",
    }

    def __init__(self, url):
        self.sparql = SPARQLWrapper(url)


    prefix = ''.join([f"PREFIX {ns}: <{uri}>\n" for ns, uri in prefixes.items()])
            

    @staticmethod
    def strip_namespace(s):
        for prefix, url in SPARQLDataProviders.prefixes.items():
            if s.startswith(url):
                return prefix + ":" + strip_prefix(url, s)
        return s

    def run_query(self, query, placeholders={}, keep_namespaces=False):
        generated = self.prefix + query
        for key, value in placeholders.items():
            generated = generated.replace("?"+key, value)
        
        self.sparql.setQuery(generated)
        self.sparql.setReturnFormat(JSON)
        results =  self.sparql.query().convert()

        res_df = pd.json_normalize(results['results']['bindings'])
        if keep_namespaces:
            value_columns = res_df.columns[res_df.columns.str.endswith("value")].to_list()

            res_df = res_df[value_columns].applymap(SPARQLDataProviders.strip_namespace)
            res_df.rename(lambda s: remove_suffix(s, ".value"), axis=1, inplace=True)
        return res_df


class DBPediaQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to DBPedia"""
    def __init__(self):
        super().__init__("http://dbpedia.org/sparql/")
    

class WikidataQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to Wikidata"""
    def __init__(self):
        super().__init__("https://query.wikidata.org/sparql")


class FusekiQuery(SPARQLDataProviders):
    def __init__(self, flavour="sample_1000_simple"):
        super().__init__(f"http://knowledge-glue-fuseki-jeffstudentsproject.ida.dcs.gla.ac.uk/{flavour}/sparql")


dbpedia_sparql = DBPediaQuery()
wikidata_sparql = WikidataQuery()
fuseki_sparql = FusekiQuery()