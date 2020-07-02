from abc import ABC, abstractmethod
from .dumps import is_file, download_to
from os.path import join

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
    def get_dump_url(self, entity, format, *args, **kwargs):
        url = f"https://wikidata.org/wiki/Special:EntityData/{entity}.{format}"
        if "flavor" in kwargs:
            url += f"?flavor={kwargs['flavor']}"
        return url
        
    def get_filename_path(self, entity, format):
        return join("wikidata", entity + "." + format)

    
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
        return join("dbpedia", entity + "." + format)