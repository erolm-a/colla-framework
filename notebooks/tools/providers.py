from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import json
from os.path import join
import pandas as pd

from .sparql_wrapper import WikidataQuery

from .dumps import is_file, download_to, wrap_open

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


class WiktionaryProvider(DataSourceProvider):
    def get_dump_url(self, entity, format, *args, **kwargs):
        """"""
        if format != "json":
            raise Exception("Unsupported non-json formats")
        return f"https://en.wiktionary.org/api/rest_v1/page/definition/{entity}"
    
    def get_filename_path(self, entity, format):
        if format != "json":
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
        exploded_columns = response_pd["definitions"].apply(pd.Series).drop("parsedExamples", axis=1)
        
        WiktionaryProvider._replace_nan(exploded_columns, "examples")

        
        return pd.concat([response_pd['partOfSpeech'], exploded_columns], axis=1)
    
    def fetch_by_label(self, label: str, format: str, language="en", *args, **kwargs):
        self.fetch_dataset(label, format, *args, **kwargs)
        with wrap_open(self.get_filename_path(label, "json")) as fp:
            result = json.load(fp)

        print(result)
        result_df = self._response_to_df(result, language)

        result_df["strippedDefinition"] = result_df["definition"].apply(
            lambda definition: BeautifulSoup(definition) \
                                .get_text())
        return result_df
