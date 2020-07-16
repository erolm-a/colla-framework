"""Some data sources (DBPedia, Wikidata) provide a SPARQL endpoint. However, their ontologies differ radically.

Here we provide some tools to simplify queries by providing proper prefixes."""

from abc import ABC, abstractmethod
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, JSONLD

class SPARQLDataProviders(ABC):
    @property
    @abstractmethod
    def prefix(self):
        pass

    def run_query(self, query, placeholders={}):
        generated = self.prefix + query
        for key, value in placeholders.items():
            generated = generated.replace("?"+key, value)
        
        self.sparql.setQuery(generated)
        self.sparql.setReturnFormat(JSON)
        results =  self.sparql.query().convert()

        return pd.json_normalize(results['results']['bindings'])

class DBPediaQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to DBPedia"""
    def __init__(self):
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql/")
    
    @property
    def prefix(self):
        return """
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX dc: <http://purl.org/dc/elements/1.1/>
            PREFIX : <http://dbpedia.org/resource/>
            PREFIX dbpedia2: <http://dbpedia.org/property/>
            PREFIX dbpedia: <http://dbpedia.org/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                """
    

class WikidataQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to Wikidata"""
    def __init__(self):
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        
    @property
    def prefix(self):
        return ""

# Export those two globally; their usage is deprecated.
dbpedia_sparql = DBPediaQuery()
wikidata_sparql = WikidataQuery()