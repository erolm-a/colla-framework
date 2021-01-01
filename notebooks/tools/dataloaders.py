'''
Convenience tools for parsing CBOR and BIO stuff

TODO: add BIO code from the notebook
'''

from trec_car import read_data
from trec_car.read_data import (AnnotationsFile, ParagraphsFile, Page, Para,
                                Section, List as ParaList, ParaLink, ParaText, ParaBody)

from .dumps import wrap_open, get_filename_path
from collections import Counter, defaultdict

import cbor
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences

import bisect
import concurrent.futures as futures
from io import BytesIO#, StringIO
import itertools
import mmap
import os
import pickle
import random
import subprocess

from deprecated import deprecated

import tokenizer_cereal

from typing import List, Union, Iterable, Tuple

def b2i(x):
    return int.from_bytes(x, "big")

class WikipediaCBOR(Dataset):
    """
    Dataset for Wikipedia loader.

    This loader is based on the CBOR dataset for the trec-car project.
    I completely dislike it. Its format makes it very hard to perform our
    question answering tasks, e.g. extracting wikipedia links, without
    tokenizing the whole text.

    Thus, this class aims to make a bridge between the trec-car mindset and an
    application-ready torch dataloader to be used inside BERT.
    """

    def __init__(self,
                 cbor_path: str,
                 partition_path: str,
                 max_entity_num=10_000,
                 num_partitions=None,
                 page_lim=None,
                 token_length=768,
                 clean_cache=False,
                 ):
        """
        :param cbor_path the relative path of the wikipedia cbor export
        :param partition_path the relative path of the partitions to use
        :param max_entity_num the maximum number of allowed entities.
        Only the top `max_entity_num` most frequent links are considered
        (ties not broken, thus the actual number will be bigger)
        :param num_partitions the number of partitions to use
        :param page_lim the number of pages in a partition
        :param token_length return only the first `token_length` tokens of a page.
        :param clean_cache delete the old cache
        """
        
        self.cbor_path = get_filename_path(cbor_path)
        self.partition_path = get_filename_path(partition_path)
        
        self.max_entity_num = max_entity_num
        self.token_length = token_length
        
        # Download vocabulary from S3 and cache.
        #self.tokenizer = torch.hub.load('huggingface/pytorch-transformers',
        #                                'tokenizer', 'bert-base-uncased')
        
        cache_path = os.path.split(self.cbor_path)[0]
        key_file = os.path.join(cache_path, "sorted_keys.pkl")
        
        offset_list_path = os.path.join(self.partition_path,
                                        "partition_offsets.txt")

        with wrap_open(offset_list_path) as f:
            self.partition_offsets = [int(x) for x in f.readlines()]
        

        if os.path.isfile(key_file) and not clean_cache:
            with open(key_file, "rb") as pickle_cache:
                # NOTE: total freqs were not saved
                self.offsets, self.key_titles, self.key_encoder, \
                    self.valid_keys = pickle.load(pickle_cache)
            tqdm.tqdm.write("Loaded from cache")
        else:
            tqdm.tqdm.write("Generating key cache")
            
            self.cbor_toc_annotations = AnnotationsFile(self.cbor_path)
            self.offsets = np.array(list(self.cbor_toc_annotations.toc.values()))
            self.offsets.sort()

            self.key_titles = self.extract_readable_key_titles()
            self.key_encoder = dict(zip(self.key_titles, itertools.count()))
            self.key_encoder["PAD"] = 0 # useful for batch transforming stuff

            # FIXME
            self.valid_keys = set(self.key_encoder.values())
            # self.key_decoder = dict(zip(self.key_encoder.keys(), self.key_encoder.values()))

            # preprocess and find the top k unique wikipedia links
            """
            self.total_freqs = self.extract_links(num_partitions, partition_page_lim=page_lim)
            threshold_value = torch.kthvalue(self.total_freqs,
                            len(self.total_freqs) - (max_entity_num - 1))[0]
            self.valid_keys = set(torch.nonzero(
                self.total_freqs >= threshold_value).squeeze().tolist())
             """
            
            with open(key_file, "wb") as pickle_cache:
                pickle.dump((self.offsets, self.key_titles, self.key_encoder,
                             self.valid_keys), pickle_cache)

        # map the original "sparser" keys to a smaller set - as long as token_length in theory
        #self.key_restrictor = dict(zip(self.valid_keys, range(self.max_entity_num)))
    
       
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
                
        key_titles = set()

        # If reloaded for a second time, this should be way faster.
        with mmap.mmap(self.cbor_toc_annotations.cbor.fileno(), 0, mmap.MAP_PRIVATE) as cbor_file:
            for offset in tqdm.tqdm(self.offsets, desc="Extracting human-readable page titles"):
                key_titles.add(extract_from_key(offset))
                
        return key_titles
     
    def __len__(self):
        # TODO: change this into the number of blocks
        return len(self.offsets)

    
    def get_attention_mask(self, tokens: List[int], p=0.2):
        """
        Get an attention mask. Padding tokens are automatically masked out.
        The others have a probability of being masked given by p.

        :param tokens the tokens to process
        :param p the dropout rate. p = 0 means no elements are excluded.
        """
        
        return [np.random.choice([0, float(tok != 0)],
                                  p=(p, 1.0-p)) for tok in tokens]
    
    def key2partition(self, key):
        """
        Find the partition that possesses the given key
        """
        partition_id = bisect.bisect_right(self.partition_offsets, key) - 1
        assert partition_id >= 0, "Keys in the CBOR header section not allowed"
        offset = key - self.partition_offsets[partition_id]

        return (partition_id, offset)
    
    def preprocess_page(self, page: Page):
        """
        Transform a list of pages into a flattened representation that can
        then be easily (de)serialized.
        """

        # For the sake of easy link spans they are byte oriented to make
        # it easier for the rust std
        page_content = BytesIO()
        links = []

        # Encode a link. Cast to padding if the link was not "common".
        # Call this method only after preprocessing has been done!
        def encode_link(link):
            #return self.key_restrictor.get(self.key_encoder.get(link, 0), 0)
            return self.key_encoder.get(link, 0)

        # People say pattern matching is overrated.
        # I beg to differ.
        # (It's also true that a tree structure for tokenization makes
        # absolutely no sense - but I don't get to decide things apparently).
        def handle_section(skel: Section):
            for subskel in skel.children:
                visit_section(subskel)

        def handle_list(skel: ParaList):
            visit_section(skel.body)

        def handle_para(skel: Para):
            paragraph = skel.paragraph
            bodies = paragraph.bodies

            for body in bodies:
                visit_section(body)

        def handle_paratext(body: ParaBody):
            page_content.write(body.get_text().encode())

        def handle_paralink(body: ParaLink):
            encoded_link = encode_link(body.page)
            start_byte_span = page_content.tell()
            end_byte_span = start_byte_span + len(body.get_text().encode()) - 1
            page_content.write(body.get_text().encode())

            links.append((encoded_link, start_byte_span, end_byte_span ))

    
        def nothing():
            return lambda body: None

        handler = defaultdict(nothing, {Section: handle_section,
                            Para: handle_para,
                            List: handle_list,
                            ParaLink: handle_paralink,
                            ParaText: handle_paratext})

        def visit_section(skel):
            # Recur on the sections
            handler[type(skel)](skel)

        for skel in page.skeleton:
            visit_section(skel)
        
        return {"text": page_content.getvalue(), "link_mentions": links}

    def preprocess(self, limit=-1):
        rust_cbor_path = self.partition_path + "/test_rust.cbor"
        rust_cereal_path = self.partition_path + "/test_rust.cereal"

        if limit == -1:
            limit = len(self)

        with open(rust_cbor_path, "wb") as fp:
            offsets = []

            with open(self.cbor_path, "rb") as cbor_fp:
                for i, page in enumerate(tqdm.tqdm(read_data.iter_annotations(cbor_fp),
                                                    total=limit)):
                    if i >= limit:
                        break

                    offsets.append(fp.tell())

                    parsed = self.preprocess_page(page)

                    # enforce this key order
                    parsed = {"id": i, **parsed}

                    cbor.dump(parsed, fp)
            
        # save cumulated block sizes
        blocks_per_page = tokenizer_cereal.tokenize_from_cbor_list(rust_cbor_path, rust_cereal_path, offsets, self.token_length)

        with open(self.partition_path + "/cumulated_block_sizes.npy", "wb") as fp:
            cumulated = np.cumsum(blocks_per_page)
            np.save(fp, cumulated)
    
    def count_frequency(self):
        """
        Use rust to count the frequency, and *hope* and pray really hard that it's fast.
        """
        rust_cereal_path = self.partition_path + "/test_rust.cereal"

        return tokenizer_cereal.count_frequency(rust_cereal_path)

    
    def __getitem__(self, idx: int):
        """
        Return a tensor with the given batches. The shape of a batch is
        (b x 3 x MAX_LEN).
        
        The first row is the token embedding via WordPiece ids.
        The second row is the BERT attention mask.
        The third row is the expected Entity Linking output.

        Given that we work with token blocks, we convert the provided indices
        to the correct page and block offsets.

        :param idx the index of the block to fetch.
        """
 
        if type(idx) != int:
            idx = idx[0]

        slice_file = self.partition_path + "/test_rust.cereal"

        with open(self.partition_path + "/cumulated_block_sizes.npy", "rb") as fp:
            cumulated = np.load(fp)

        page_idx = bisect.bisect_right(cumulated, idx)
        page_block_nums = cumulated[page_idx] - (cumulated[page_idx-1] if page_idx > 0 else 0)
        block_offset = idx - (cumulated[page_idx-1] if page_idx > 0 else 0)

        # Too tired to properly handle this edge case
        if page_block_nums == 0:
            page_idx = 0
            block_offset = 0
        
        toks, links = tokenizer_cereal.get_token_slice(slice_file, page_idx, block_offset, self.token_length)

        toks = pad_sequences([toks], maxlen=self.token_length, dtype="long",
                                value=0.0, truncating="post", padding="post")[0]

        links = pad_sequences([links], maxlen=self.token_length, dtype="long",
                                value=0.0, truncating="post", padding="post")[0]
        
        attns = self.get_attention_mask(toks)

        # TODO: possibly change the float format to float16...
        # Does the change happen when moving to GPU?
        # Need to investigate this...
        toks_list = torch.LongTensor(toks).squeeze()
        attns_list = torch.FloatTensor(attns).squeeze()
        links_list = torch.LongTensor(links).squeeze()

        return (toks_list, attns_list, links_list)


class BIO:
    """
    Load and process the GBK corpus
    """
    def __init__(self, dataset_dir, token_length, ):
        self.token_length = token_length
        # Download tokenizer from S3 and cache.
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                                        'tokenizer', 'bert-base-uncased')
        data_pd = self.__load_dataset(dataset_dir)
        self.__simplify_bio(data_pd)
        utterances, labels = self.__aggregate(data_pd)

        bio_values = list(set(data_pd["bio"].values))
        bio_values.append("PAD")

        # BIO tag to a numerical index (yes, this is dumb way to make an enum)
        # Apparently one row is misclassified as p?
        self.bio2idx = {t: i for i, t in enumerate(bio_values)}
        self.bio2idx['p'] = self.bio2idx['O']

        # Tokenize and split
        tokenized_texts_labels = [
           self.__tokenize_preserve_labels(sent, labs) for sent, labs in
                zip(utterances, labels)
        ]

        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_labels]
        tokenized_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_labels]

        # Pad and convert to ids...
        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt)
                                            for txt in tokenized_texts],
                                        maxlen=self.token_length, dtype="long",
                                        value=0.0, truncating="post",
                                        padding="post")

        self.labels = pad_sequences([[self.bio2idx.get(l) for l in lab]
                                            for lab in tokenized_labels],
                                        maxlen=self.token_length,
                                        value=self.bio2idx["PAD"],
                                        padding="post", dtype="long",
                                        truncating="post")

        self.attention_mask = [[float(i != 0.0) for i in ii] for ii in self.input_ids]

        # Tensorize...
        self.input_ids = torch.tensor(self.input_ids)
        self.labels = torch.tensor(self.labels)
        self.attention_mask = torch.tensor(self.attention_mask)


    def __load_dataset(self, dataset_dir):
        """
        Load the dataset
        """
        names = []
        with wrap_open("ner.csv", "r", encoding="latin1") as f:
            names = ["index"] + f.readline().strip().split(",")[1:]
            names = names + list(range(34 - len(names)))

        with wrap_open("ner.csv", "rb") as f:
            f.readline() # skip the first line
            data = pd.read_csv(f, encoding="latin1", names=names) \
                            .fillna(method="ffill")
        return data
    

    @staticmethod
    def __simplify_bio(data_pd: pd.DataFrame):
        """
        Add another column called "bio" that is a simplification of the original tagging value.
        We thus exclude the various nuances of the classification (entity, building, city etc.).
        """
        def helper(column): return column[0]
        data_pd["bio"] = data_pd["tag"].apply(helper)


    @staticmethod
    def __aggregate(data_pd: pd.DataFrame):
        """
        Group the sentences by their sentence_idx.

        Return utterances (list of words) and labels (list of whatever)
        """

        # only extract word and bio when grouping
        def helper(s):
            return [(w, t) for w, t in zip(s["word"].values.tolist(),
                                           s["bio"].values.tolist())]

        sentences = [s for s in data_pd.groupby("sentence_idx").apply(helper)]
        utterances = [[w[0] for w in s] for s in sentences]
        labels = [[w[1] for w in s] for s in sentences]

    
        return utterances, labels

    def __tokenize_preserve_labels(self, sentence, text_labels):
        """
        Tokenize the given sentence. Extend the corresponding label
        for all the tokens the word is made of.
        
        Assumption: len(sentence) == len(text_labels)
        """
        
        tokenized_sentence = []
        labels = []
        
        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.extract_links_only(word)
            n_subwords = len(tokenized_word)
            
            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)
        
        return tokenized_sentence, labels
    
    def get_pytorch_dataset(self) -> Dataset:
        """
        Return the whole dataset as a TensorDataset. 
        """
        return TensorDataset(self.input_ids, self.attention_mask, self.labels)