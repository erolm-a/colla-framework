"""
This module provides some convenience functions for when downloading
or fetching data dumps. When cloning this repo, the data folder will be
initially empty. This module helps populating the folder without having
to manually create subfolders or making unrequired pollution.
"""

import os
import requests
import wget

from .strings import strip_prefix

DATA_FOLDER = "data"


def get_filename_path(filename):
    """
    Prefix a filename with the DATA_FOLDER.
    Also create all the base folders.
    """
    filename = os.path.join(DATA_FOLDER, filename)
    basefolder = os.path.split(filename)[0]
    os.makedirs(basefolder, exist_ok=True)

    return filename

def wrap_open(filename, *args, **kwargs):
    """Wrapper to python's open() that prepends DATA_FOLDER.
    
    If a user provides a open-shaped `handler` as kwarg it will be used
    instead.
    """
    path = get_filename_path(filename)

    if "handler" in args:
        handler = kwargs["handler"]
        del kwargs["handler"]
    else:
        handler = open
    return handler(path, *args, **kwargs)


def is_file(filename):
    """Wrapper for os.path.isfile that automatically prefixes a filename"""
    return os.path.isfile(get_filename_path(filename))


def download_to(url: str, filename: str):
    """
    Wrapper to wget that prefixes the destination file and possibly creates
    all the required subfolders
    """
    # touch and create the directories
    path = get_filename_path(filename)
    # print(url)
    wget.download(url, get_filename_path(filename), bar=wget.bar_thermometer)
