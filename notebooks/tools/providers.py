from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import json
from os.path import join
import pandas as pd

from .sparql_wrapper import WikidataQuery, FusekiQuery
from .dumps import is_file, download_to, wrap_open, get_filename_path
import tempfile


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
    """Provider for Wikidata"""
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

    def dump_grammatical_categories(self):
        """
        Dump a DataFrame of Grammatical categories.
        """
        grammatical_categories_file = self.get_filename_path("grammatical_categories", "json")
        if is_file(grammatical_categories_file):
            print("Wikidata grammatical categories downloaded; skipping")
            with wrap_open(grammatical_categories_file) as fp:
                return pd.read_json(fp)
        
        grammatical_categories_file = get_filename_path(grammatical_categories_file)

        grammatical_categories = self.sparql.run_query("""
        SELECT ?entity ?entityLabel
        WHERE
        {
            ?entity wdt:P31/wdt:P279* wd:Q980357.
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """)
        
        grammatical_categories = grammatical_categories[~grammatical_categories["entityLabel.xml:lang"].isna()][["entity.value", "entityLabel.value"]]
        grammatical_categories.to_json(grammatical_categories_file)
        return grammatical_categories
    
    def dump_pos_categories(self) -> pd.DataFrame:
        """
        Dump a DataFrame of POS categories.
        """
        pos_categories_file = self.get_filename_path("pos_categories", "json")

        if is_file(pos_categories_file):
            print("Wikidata POS categories downloaded: skipping")
            with wrap_open(pos_categories_file) as fp:
                return pd.read_json(fp)
        
        pos_categories_file = get_filename_path(pos_categories_file)
        
        pos_categories = self.sparql.run_query("""
        SELECT ?entity ?entityLabel
        WHERE
        {
        ?entity wdt:P31 wd:Q82042.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """)

        pos_categories = pos_categories[~pos_categories["entityLabel.xml:lang"].isna()][["entity.value", "entityLabel.value"]]
        pos_categories.to_json(pos_categories_file)
        return pos_categories

    
class DBPediaProvider(DataSourceProvider):
    """Provider for DBPedia"""
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
    """
    Provider for the RESTFul API of Wiktionary.

    The API endpoint is available here: 
    https://en.wiktionary.org/api/rest_v1

    Note: its usage is deprecated as a small amount of information is indeed
    available. Users should use the  wiktextract pipeline to extract
    definitions, or `FusekiProvider` for ready-made results.
    """
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
        dump_name = f"enwiktionary-{revision}-pages-{variant}.xml.{format}"
        basefile = f"wiktionary/{dump_name}"
        url = f"https://dumps.wikimedia.org/enwiktionary/{revision}/{dump_name}"
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
        """Extract a single entity.
        """
        # TODO this call has a different format from the others!
        query = """
        CONSTRUCT
        WHERE
        {
            ?entity ?p ?o.
        }"""
        rdflib_graph = self.fuseki_sparql.run_query(query, placeholders={"entity": entity})
        return json.loads(rdflib_graph.serialize(format=format))

    def get_filename_path(self, entity, format):
        raise Exception("Not available")

    def fetch_by_label(self, label, format=None, *args, **kwargs):
        result = self.fuseki_sparql.run_query("""
            SELECT ?entity ?pos ?sense ?example ?senseDefinition
            WHERE
            {
                ?entity kglprop:label "?label"@en;
                        kglprop:sense ?sense;
                        kglprop:pos ?pos.
                ?sense kglprop:definition ?senseDefinition.
                OPTIONAL {?sense kglprop:example ?example. }
                OPTIONAL {?sense kglprop:subsense/kglprop:example ?example. }
                OPTIONAL {?sense kglprop:subsense/kglprop:usage/kglprop:example ?example. }
            }
            """, {'label': label}, True)
        
        if len(result) == 0:
            return None

        if not "example" in result.columns:
            result['example'] = ""

        # Group examples by senses
        distinct_senses = result.drop("example", axis=1).drop_duplicates()
        examples_grouped = result.groupby("sense")['example'].apply(list).reset_index(name="examples")
        return examples_grouped.join(distinct_senses.set_index("sense"), on="sense")

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

    def fetch_usages(self, sense_id: str, flatten=True):
        """
        Find usages for a given sense.

        sense_id is the id of a sense in the form: kgl:id-Sx.

        If flatten is true, return both subsenses and usages.
        Otherwise, return only the subsenses.
        """
        if flatten:
            query = """
            SELECT ?description
            WHERE
            {
              {
                ?sense_id kglprop:subsense/kglprop:usage/kglprop:definition ?description.
              }
              UNION
             {
                ?sense_id kglprop:subsense/kglprop:definition ?description.
              }
            }
            """
        else:
            query = """
            SELECT ?description
            WHERE
            {
                ?sense_id kglprop:subsense/kglprop:definition ?description.
            }
            """
        return self.fuseki_sparql.run_query(query, placeholders={"sense_id": sense_id})

    
    def fetch_forms(self, lexeme_id, feature):
        """
        Search a form by some criteria.
        
        Available criteria:
            - label: the lexeme (root form) must match this
            - feature: a grammatical feature must match.
        """
        query = """
        SELECT ?form ?formLabel
        WHERE
        {
            ?lexeme_id kglprop:form ?form.
            ?form kglprop:label ?formLabel.
            ?form kglprop:grammaticalFeature/kglprop:label ?feature.
        }
        """
        return self.fuseki_sparql.run_query(query, placeholders={"lexeme_id": lexeme_id,
                                                                 "feature": feature})

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


    def dump_pos_categories(self):
        """
        Dump the POS categories.
        POS entities in the KG are guaranteed to have human-readable names.
        Thus, one can easily remove the kgl prefix to obtain standalone
        labels.
        """
        query = """
        SELECT ?posEntity
        WHERE
        {
            ?posEntity a kgl:POS.
        }
        """
        return self.fuseki_sparql.run_query(query, keep_namespaces=True)
        
        
    @staticmethod
    def dump_full_dataset(self, format, flavour, *args, **kwargs):
        if format != "ttl":
            raise Exception("Unsupported format " + format)
        basefile = f"fuseki/dump-{flavour}.ttl"
        url = f"http://knowledge-glue-fuseki-jeffstudentsproject.ida.dcs.gla.ac.uk/{flavour}/data"
        download_to(url, basefile)

################ Word lists #################

class Wordlist(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_wordlist():
        """
        Get a list of words
        """
        pass

def scrape_wiktionary_wordlists(url: str, name: str):
    filename = "wordlists/" + name
    if not is_file(filename):
        download_to(url, filename + ".html")
        # TODO: make this file temporary
        with wrap_open(filename + ".html") as fp:
            parsed = BeautifulSoup(fp.read(), "html.parser")
            tables = parsed.find_all("table")
            words = []
            for table in tables:
                rows = table.find_all("tr")[1:]
                cols = [row.find_all("td")[1].find("a").text for row in rows]
                words.extend(cols)
            with wrap_open(filename, "w") as save_fp:
                save_fp.writelines([word + "\n" for word in words])
                
        return cols
    else:
        with wrap_open(filename) as fp:
            return [word.strip() for word in fp.readlines()]


class WiktionaryTV(Wordlist):
    """
    Extract from the list of 1,000 most common words in TV scripts between 2000 and 2006.
    For a more complete explanation on how this data was harvested see here:

    https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/TV/2006/explanation
    """

    @staticmethod
    def get_wordlist():
        return scrape_wiktionary_wordlists("https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/TV/2006/1-1000", "wdtv.txt")


class WiktionaryProjectGutenberg(Wordlist):
    """
    Extract from the list of 10.000 common words of the Project Gutenberg
    """
    @staticmethod
    def get_wordlist():
        return scrape_wiktionary_wordlists("https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/PG/2006/04/1-10000", "wdpg.txt")


