import os
import requests

from .strings import strip_prefix

DATA_FOLDER = "data"


def get_filename_path(filename):
    return os.path.join(DATA_FOLDER, filename)


def wrap_open(filename, *args, **kwargs):
    """wrapper to python's open() that prepends DATA_FOLDER"""
    path = get_filename_path(filename)
    basefolder = os.path.join(DATA_FOLDER, os.path.split(filename)[0])
    os.makedirs(basefolder, exist_ok=True)

    if "handler" in args:
        handler = args["handler"]
    else:
        handler = open
    return handler(path, *args, **kwargs)


def is_file(filename):
    return os.path.isfile(get_filename_path(filename))


def download_to(url: str, filename: str):
    r = requests.get(url)
    with wrap_open(filename, "wb") as f:
        f.write(r.content)
