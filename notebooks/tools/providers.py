from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import json
from os.path import join
import pandas as pd

from .sparql_wrapper import WikidataQuery, FusekiQuery

from .dumps import is_file, download_to, wrap_open
from .strings import strip_prefix

class DataSourceProvider(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_dump_url(self, entity, format, *args, **kwargs):
        """Get a URL dumper for the data source. In general, we assume that this
        is how 'any' conventional dataset works, but some may differ - see
        Google datasets requiring gsutils.
        
        If a user wants to perform something more sophisticated one
        could consider using a SPARQLWrapper."""
        pass
    
    @abstractmethod
    def get_filename_path(self, entity, format):
        pass

    @abstractmethod
    def fetch_by_label(self, label, format, *args, **kwargs):
        """Fetch one or more definitions given a label.
           Implementers should return a `list` of senses or dumps,
           whatever this may mean for the underlying provider.
        """
        pass

    @staticmethod
    @abstractmethod
    def dump_full_dataset(format, revision, *args, **kwargs):
        """Dump a full dataset. The dump is meant to be used by the users
           rather than the provider itself, for example because it does not
           have indices or is not in a suitable format.
        """
        pass
    
    def fetch_dataset(self, entity, format, force_redownload=False, 
                      *args, **kwargs):
        url = self.get_dump_url(entity, format, *args, **kwargs)
        filename = self.get_filename_path(entity, format)
    
        if is_file(filename) and not force_redownload:
            print(f"Dataset {filename} already downloaded. Skipping...")
        else:
            print(f"Downloading {filename} from {url}")
            download_to(url, filename)
            print("Done")


class WikidataProvider(DataSourceProvider):
    def __init__(self):
        self.sparql = WikidataQuery()

    def get_dump_url(self, entity, format, *args, **kwargs):
        url = f"https://wikidata.org/wiki/Special:EntityData/{entity}.{format}"
        if "flavor" in kwargs:
            url += f"?flavor={kwargs['flavor']}"
        return url
        
    def get_filename_path(self, entity, format):
        return join("wikidata", entity + "." + format)

    def fetch_by_label(self, label, format, *args, **kwargs):
        entities = self.sparql.run_query("""
            SELECT ?entity
            WHERE
            {
                ?entity rdfs:label|skos:altLabel "?label"@en.
            }
        """, {"label", label})["entity.value"]
        return [self.fetch_dataset(entity, "json", *args, **kwargs) for entity in entities]

    @staticmethod
    def dump_full_dataset(format, revision, *args, **kwargs):
        raise Exception("Not implemented yet")

    
class DBPediaProvider(DataSourceProvider):
    def get_dump_url(self, entity, format, *args, **kwargs):
        """Extract a dbpedia entity. Very simple and crude. More sofisticated
        queries should use a SPARQLWrapper.
    
        Note that in DBPedia wikipedia links are entity identifiers and are
        case-sensitive. The URL fetcher seems *not* to be solving the redirects
        alone.
        """
        return f"https://dbpedia.org/data/{entity}.{format}"

    def get_filename_path(self, entity, format):
        return join("dbpedia", f"{entity}.{format}")

    def fetch_by_label(self, label, format, *args, **kwargs):
        # TODO: implement disambiguation management
        raise Exception("DBPedia LIKE search not yet implemented")

    @staticmethod
    def dump_full_dataset(self, format, revision, *args, **kwargs):
        raise Exception("Not implemented yet")


class WiktionaryProvider(DataSourceProvider):
    def get_dump_url(self, entity, format, *args, **kwargs):
        if format != "json":
            raise Exception("Unsupported non-json formats")
        return f"https://en.wiktionary.org/api/rest_v1/page/definition/{entity}"
    
    def get_filename_path(self, entity, format):
        if format not in ["json", "parquet"]:
            raise Exception("Unsupported non-json formats")
        return join("wiktionary", f"{entity}.{format}")

    @staticmethod
    def _replace_nan(df, column):
        """Replace NaNs in a column with an empty list.
        
        Required because pd.Series.fillna() does not accept lists.
        See https://stackoverflow.com/a/61944174
        """
        is_nan = df[column].isna()
        to_replace = pd.Series([[]] * is_nan.sum()).values
        df.loc[is_nan, column] = to_replace

    @staticmethod        
    def _response_to_df(response, language="en"):
        """explode definitions and convert the inner dicts into a pandas series, then join with PoS"""
        response_pd = pd.json_normalize(response[language]).explode("definitions")
        exploded_columns = response_pd["definitions"].apply(pd.Series).drop("parsedExamples", axis=1, errors='ignore')
        
        WiktionaryProvider._replace_nan(exploded_columns, "examples")

        
        return pd.concat([response_pd['partOfSpeech'], exploded_columns], axis=1)
    
    def fetch_by_label(self, label: str, format: str, language="en", *args, **kwargs):
        self.fetch_dataset(label, format, *args, **kwargs)
        with wrap_open(self.get_filename_path(label, "json")) as fp:
            result = json.load(fp)

        result_df = self._response_to_df(result, language)

        result_df["strippedDefinition"] = result_df["definition"].apply(
            lambda definition: BeautifulSoup(definition) \
                                .get_text())
        return result_df

    @staticmethod
    def dump_full_dataset(format="bz2", revision="latest", variant="articles"):
        """Download a Wiktionary dump.

        params:
        - `revision`: a date represented as a `str` in the format YYYYMMDD        
        - `format`: a compression format; the user is expected to know in advance which format is needed.
        - `variant`: by default dump the article dataset.
        
        """
        basefile = f"wiktionary/enwiktionary-{revision}-pages-{variant}.xml.{format}"
        url = f"https://dumps.wikimedia.org/enwiktionary/{revision}/{basefile}"
        download_to(url, basefile)

class FusekiProvider(DataSourceProvider):
    def __init__(self, flavour="sample_10000_common"):
        super().__init__()
        self._flavour = flavour
        self._fuseki_sparql = FusekiQuery(flavour)

    @property
    def fuseki_sparql(self):
        return self._fuseki_sparql

    @property
    def flavour(self):
        return self._flavour
    
    def get_dump_url(self, entity, format, *args, **kwargs):
        """Extract a single entity. Currently not implemented
    
        Note that in DBPedia wikipedia links are entity identifiers and are
        case-sensitive. The URL fetcher seems *not* to be solving the redirects
        alone.
        """
        raise Exception("Not available")

    def get_filename_path(self, entity, format):
        raise Exception("Not available")

    def fetch_by_label(self, label, format, *args, **kwargs):
        return self.fuseki_sparql.run_query("""
            SELECT ?entity ?pos ?sense ?example ?related ?senseDefinition
            WHERE
            {
                ?entity kglprop:label "?label"@en;
                        kglprop:sense ?sense;
                        kglprop:pos ?pos.
                ?sense kglprop:definition ?senseDefinition.
                OPTIONAL {?sense kglprop:example ?example. }
                OPTIONAL {?sense kglprop:related ?related. }
            }
            """, {'label': label}, True)

    def fetch_examples(self, sense, *args, **kwargs):
        """Fetch an example for a given sense.
        
        Sense must be an instantiation of a kgl
        in the form of kgl:id-S0

        Returns a dataframe with one column, "example",
        containing examples (if any).
        """
        return self.fuseki_sparql.run_query("""
            SELECT ?example
            WHERE
            {
                ?sense kglprop:example ?example
            }
        """, {'sense': sense}, True)
    
    def fetch_forms(self, label=None, pos=None):
        """
        Search a form by some criteria.
        
        Available criteria:
            - label: the lexeme (root form) must match this
            - pos: the part of speech must match
            - feature: a grammatical feature must match.
        """
        pass

    def fetch_all_grammatical_categories(self, language="en"):
        """
        Return a list of all available grammatical categories for the given
        language
        """

        query = """
        SELECT ?grammaticalCategoryEntity ?grammaticalCategoryLabel
        WHERE
        {
            ?grammaticalCategoryEntity a kgl:GrammaticalCategory;
                                            rdfs:label ?grammaticalCategoryLabel.
        }
        """
        return self.fuseki_sparql.run_query(query)['grammaticalCategoryLabel.value']

    @staticmethod
    def dump_full_dataset(self, format, flavour, *args, **kwargs):
        if format != "ttl":
            raise Exception("Unsupported format " + format)
        basefile = f"fuseki/dump-{flavour}.ttl"
        url = f"http://knowledge-glue-fuseki-jeffstudentsproject.ida.dcs.gla.ac.uk/{flavour}/data"
        download_to(url, basefile)