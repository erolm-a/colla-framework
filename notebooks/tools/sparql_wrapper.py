"""Some data sources (DBPedia, Wikidata) provide a SPARQL endpoint. However, their ontologies differ radically.

Here we provide some tools to simplify queries by providing proper prefixes."""

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, JSONLD

from .strings import strip_prefix, remove_suffix

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

    def sanitize_placeholder(self, placeholder: str) -> str:
        # TODO: should this method be private and/or static
        # First, check if the placeholder is not a free variable
        if placeholder.startswith("?"):
            raise Exception("Detected free variable here")
        # Then, check if the placeholder is not trying to break
        # free from quotes.
        if '"' in placeholder or "'" in placeholder:
            raise Exception("Quoted values not allowed")
        return placeholder


    def run_query(self, query, placeholders={}, keep_namespaces=False) -> pd.DataFrame:
        """
        Run a SPARQL query.
        This method makes sure to properly set the query, the return type,
        perform substitution and possibly escaping

        :param query: the query to run. The query can contain placeholders in
        the form of `?placeholder`. Placeholders can be quoted in strings, and
        will be substituted regardless if provided in the `placeholders` param.
        :param placeholders: a dictionary of placeholder substitutions to perform.
        For security reasons the mapped value must be either a quoted string or
        an entity.
        :param keep_namespaces: if True, only keep the value columns and replace their
        urls with namespaces.
        :return: for SELECT queries, a pandas dataframe of triples.

        If the required columns are `foo`, `bar` etc. and keep_namespace is False
        the dataframe will contain the columns `foo.type`, `foo.value`, `bar.type`,
        `bar.value` etc.; if keep_namespace is false, the `.type` columns will be
        deleted and the `.value` columns will be stripped of their suffix.

        For CONSTRUCT queries, just return a ConjunctiveGraph
        """
        query = query.strip()

        result_is_graph = query.lower().startswith(("describe", "construct"))

        generated = self.prefix + query
        for key, value in placeholders.items():
            sanitized = self.sanitize_placeholder(value)
            generated = generated.replace("?"+key, sanitized)
        self.sparql.setQuery(generated)

        if result_is_graph:
            self.sparql.setReturnFormat(JSONLD)
            pass
        else:
            self.sparql.setReturnFormat(JSON)

        results =  self.sparql.queryAndConvert()

        if not result_is_graph:
            res_df = pd.json_normalize(results['results']['bindings'])
            res_df.fillna("", axis=1, inplace=True)
            if keep_namespaces:
                value_columns = res_df.columns[res_df.columns.str.endswith("value")].to_list()

                res_df = res_df[value_columns].applymap(SPARQLDataProviders.strip_namespace)
                res_df.rename(lambda s: remove_suffix(s, ".value"), axis=1, inplace=True)
            return res_df
        return results


class DBPediaQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to DBPedia"""
    def __init__(self):
        super().__init__("http://dbpedia.org/sparql/")
    

class WikidataQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to Wikidata"""
    def __init__(self):
        super().__init__("https://query.wikidata.org/sparql")


class FusekiQuery(SPARQLDataProviders):
    """Wrapper to make SPARQL queries to our Fuseki provider"""
    def __init__(self, flavour="sample_10000_common"):
        super().__init__(f"http://knowledge-glue-fuseki-jeffstudentsproject.ida.dcs.gla.ac.uk/{flavour}/sparql")


dbpedia_sparql = DBPediaQuery()
wikidata_sparql = WikidataQuery()
fuseki_sparql = FusekiQuery()