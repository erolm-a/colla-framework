'''
Convenience tools for parsing CBOR
'''

from trec_car import read_data
from trec_car.read_data import AnnotationsFile, ParagraphsFile, Page

from .dumps import wrap_open, get_filename_path
from collections import defaultdict

import torch
import cbor

from typing import List, Union, Iterable
from torch.utils.data import Dataset
import tqdm
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import random

import subprocess
import os
import mmap
import concurrent.futures as futures


tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.

from trec_car.read_data import Page, Section, List, Para, ParaLink, ParaText, ParaBody

def handle_section(skel, toks, links, tokenize):
    for subskel in skel.children:
        visit_section(subskel, toks, links, tokenize)

def handle_list(skel, toks, links, tokenize):
    visit_section(skel.body, toks, links, tokenize)

def handle_para(skel: Para, toks, links, tokenize):
    paragraph = skel.paragraph
    bodies = paragraph.bodies

    for body in bodies:
        visit_section(body, toks, links, tokenize)

def handle_paratext(body: ParaBody, toks, links, tokenize):
    if tokenize:
        lemmas = tokenizer.tokenize(body.get_text())
        toks.extend(lemmas)
        links.extend(["PAD"] * len(lemmas))

def handle_paralink(body: ParaLink, toks, links, tokenize):
    lemmas = tokenizer.tokenize(body.get_text())
    if tokenize:
        toks.extend(lemmas)
        links.extend([body.page] + ["PAD"] * (len(lemmas) - 1))
    else:
        links.append(body.page)
    pass

def nothing():
    return lambda body, toks, links, tokenize: None

handler = defaultdict(nothing, {Section: handle_section,
                     Para: handle_para,
                     List: handle_list,
                     ParaLink: handle_paralink,
                     ParaText: handle_paratext})


def visit_section(skel, toks, links, tokenize=True):
    # Recur on the sections
    handler[type(skel)](skel, toks, links, tokenize)





def partition(toc: str, destination: str, num_partitions=100):
    """
    ...why am I doing this by hand?
    """
    with wrap_open(toc + ".toc", "rb") as f:
        toc_file = cbor.load(f)
    
    offsets = list(toc_file.values())
    offsets.sort()
    
    destination = get_filename_path(destination)
    os.makedirs(destination, exist_ok=True)

    # bucketize
    for num_partition, i in enumerate(tqdm.trange(0, len(offsets), (len(offsets) // num_partitions) + 1)):
        j = min(i + (len(offsets) // num_partitions) + 1, len(offsets) - 1)
        start_offset, end_offset = offsets[i], offsets[j]
        partition_fname = f"{destination}/{num_partition}.cbor"
        partition_toc_fname = f"{destination}/{num_partition}.cbor.toc"
        partition_size = end_offset - start_offset
        
        # Invoke dd and do the actual splitting
        # skip is defined in terms of the input size
        # so keep ibs high to avoid thrashing
        # use tqdm to get a nice status bar for dd (will it work?)
        proc = subprocess.Popen(["dd", f"if={get_filename_path(toc)}", f"of={partition_fname}",
                        "ibs=1", "obs=1M", f"skip={start_offset}", f"count={end_offset-start_offset}",
                        "status=progress"], stdout=subprocess.PIPE)
        
        # tqdm.tqdm.write(proc.stdout.readline().decode('utf-8'))
        
        # write the per-partition toc file
        with wrap_open(partition_toc_fname, "w") as output_toc:
            for off in offsets[i:j+1]:
                output_toc.write(str(off - start_offset) + "\n")


# Torch-related stuff
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
import tqdm
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences

from tools.dumps import get_filename_path
from trec_car.read_data import AnnotationsFile, ParagraphsFile, Page

from collections import Counter
import concurrent.futures as futures
import random
import itertools
import sys
import os

import numpy as np

from typing import List, Union, Iterable

def b2i(x):
    return int.from_bytes(x, "big")


class WikipediaCBOR(Dataset):
    """
    Dataset for Wikipedia loader.
    """
    def __init__(self,
                 cbor_path: str,
                 partition_path: str,
                 # old_wikipedia_cbor: WikipediaCBOR = None, # that's a horrible debugging hack
                 max_entity_num=1_000_000,
                 num_partitions=None):
        """
        :param cbor_path the path of the wikipedia cbor export
        :param num_partitions the number of partitions to use (DEBUG!)
        """
        # Let trec deal with that in my place
        
        self.cbor_path = get_filename_path(cbor_path)
        self.partition_path = get_filename_path(partition_path)
        
        # TODO: kept here for debugging purposes, remove this
        
        # if old_wikipedia_cbor:
        self.cbor_toc_annotations = AnnotationsFile(self.cbor_path)
        self.keys = np.fromiter(self.cbor_toc_annotations.keys(), dtype='<U64')
        # preprocess and find the top k unique wikipedia links
        self.key_titles = self.extract_readable_key_titles()
        self.key_encoder = dict(zip(self.key_titles, itertools.count()))

        # page frequencies
        self.max_entity_num = max_entity_num
        self.total_freqs = self.__extract_links(num_partitions)
        
        
    def extract_readable_key_titles(self):
        """
        Build a list of human-readable names of CBOR entries.
        Compared to self.keys, these keys are not binary encoded formats.
        """
        def extract_from_key(offset):
            cbor_file.seek(offset)

            # We refer to the RFC 8949 about the CBOR structure
            # See https://tools.ietf.org/html/rfc8949 for details
            len_first_field = b2i(cbor_file.read(1))
            field_type = (len_first_field & (0b11100000)) >> 5

            # array
            if field_type == 0b100:
                # ignore the next byte
                cbor_file.read(1)
                first_elem_header = b2i(cbor_file.read(1))
                first_elem_len = first_elem_header & 31
                # first_elem_tag = first_elem_header >> 5

                if first_elem_len > 23:
                    first_elem_len = b2i(cbor_file.read(first_elem_len - 23))

                return cbor_file.read(first_elem_len).decode('utf-8')

            else:
                raise Exception("Wrong header")
                
        # Sorted seeks should make the OS scheduler less confused, hopefully
        values = list(self.cbor_toc_annotations.toc.values())
        values.sort()
        
        key_titles = set()
        
        # If reloaded for a second time, this should be way faster.
        with mmap.mmap(self.cbor_toc_annotations.cbor.fileno(), 0, mmap.MAP_PRIVATE) as cbor_file:
            for offset in tqdm.tqdm(values, desc="Extracting human-readable page titles"):
                key_titles.add(extract_from_key(offset))
                
        return key_titles
     

    def __get_pages (self,
                     partition_id: int,
                     offset_list: Iterable[int]) -> List[Page]:
        """
        Extract all the Page's from a CBOR partition file.
        It returns the list of pages in a partition
        # TODO is this reasonable?
        
        :param partition_id the index of the partition
        :param offset_list the list of page offsets within the partition
        """
        
        pages = []
        
        with open(os.path.join(self.partition_path, f"{partition_id}.cbor"), "rb") as f:
            for offset in offset_list:
                f.seek(offset)
                pages.append(Page.from_cbor(cbor.load(f)))
                #yield Page.from_cbor(cbor.load(f))
                
        return pages

    def __extract_links_monothreaded(self,
                                     partition_id: int,
                                     offset_list: Iterable[int]) -> torch.sparse.LongTensor:
        """
        Calculate the frequency of each mention in wikipedia.
        
        :param partition_id the partition to use
        :param offset_list the list of offset within that partition
        :returns a sparse torch tensor
        """
        
        links = []
        
        for page in self.__get_pages(partition_id, offset_list):
            # remove spurious None elements
            if page is None:
                continue
            
            for skel in page.skeleton:
                visit_section(skel, [], links, False)
            
        freqs = Counter(links)

        # remove mentions that do not have an associated wikipedia page
        keys = list(freqs.keys())
        for key in keys:
            if key not in self.key_titles:
                del freqs[key]

        keys = np.array([[0, self.key_encoder.get(k, -1)] for k in freqs.keys()]).T.reshape(2, -1)
        values = np.fromiter(freqs.values(), dtype=np.int32)

        return torch.sparse_coo_tensor(keys, values,
                                             size=(1, len(self)))

    def __extract_links(self, num_partitions=None, partition_page_lim=10000, pages_per_worker=1000):
        """
        Create some page batches and count mention occurrences for each batch.
        Summate results.

        This method is threaded (hopefully for the common good).
        
        :param partition_page_lim the maximum number of pages to take in a partition (None for all)
        """
            
        if num_partitions is None:
            num_partitions = 100 # TODO: automatically determine this

       
        offset_lists = []
        for partition_id in tqdm.trange(num_partitions, desc="Reading partition tocs"):
            with open(os.path.join(self.partition_path, f"{partition_id}.cbor.toc")) as f:
                offset_lists.append([int(line.strip()) for line in f.readlines()])
            
        
        starting_tensor = torch.sparse.LongTensor(1, len(self))
        tensors = []
        with futures.ThreadPoolExecutor() as executor:
            promises = []
            
            if partition_page_lim is None:
                _partition_page_lim = len(offset_lists[partition_id])
            else:
                _partition_page_lim = partition_page_lim
            
            for partition_id in tqdm.trange(num_partitions, desc="Submitting partitions"):
                for idx in range(0, _partition_page_lim, pages_per_worker):
                    chosen_page_offsets = offset_lists[partition_id][idx:idx+pages_per_worker]

                    #promises.append(executor.submit(self.__extract_links_monothreaded,
                    #                            partition_id,
                    #                            chosen_page_offsets))
                    tensors.append(self.__extract_links_monothreaded(partition_id, chosen_page_offsets))

            #for promise in tqdm.tqdm(futures.as_completed(promises), desc="Merging tokenized text"):
            #    tensors.append(promise.result())
        
        return torch.sparse.sum(torch.stack(tensors), [1])
    
    
    def __len__(self):
        return len(self.keys)
    
    def __tokenize(self, page: Page):
        toks = []
        links = []
        for skel in page.skeleton:
            visit_section(skel, toks, links)
        return toks, links
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        elif type(idx) != list:
            idx = [idx]
        
        pages = [self.cbor_toc_annotations.get(k.encode('ascii')) for k in self.keys[idx]]
        
        # can we parallelize this?
        result = [self.__tokenize(page) for page in pages]
        print(result)
        
        return torch.tensor(result)
