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