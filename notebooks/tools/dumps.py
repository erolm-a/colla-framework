"""
This module provides some convenience functions for when downloading
or fetching data dumps. When cloning this repo, the data folder will be
initially empty. This module helps populating the folder without having
to manually create subfolders or making unrequired pollution.
"""

import os
import requests
import wget
from tqdm import tqdm
from .strings import strip_prefix

DATA_FOLDER = os.environ.get("COLLA_DATA_FOLDER", os.path.join(os.getcwd(), "data"))


def get_filename_path(filename) -> str:
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

    if "handler" in kwargs:
        print("Found custom handler ", kwargs["handler"])
        handler = kwargs["handler"]
        del kwargs["handler"]
    else:
        handler = open
    return handler(path, *args, **kwargs)


def is_file(filename) -> bool:
    """Wrapper for os.path.isfile that automatically prefixes a filename"""
    return os.path.isfile(get_filename_path(filename))


def download_to(url: str, filename: str):
    """
    Download a resource in a destination file and possibly creates
    all the required subfolders. Uses tqdm
    """
    # touch and create the directories
    path = get_filename_path(filename)

    # https://stackoverflow.com/a/37573701 , but with a better block size
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 mebibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print(f"Error while downloading the requested file: {path}")