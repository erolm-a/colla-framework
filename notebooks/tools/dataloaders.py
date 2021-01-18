'''
Convenience tools for parsing CBOR and BIO stuff

TODO: add BIO code from the notebook
'''
import bisect
from collections import defaultdict, namedtuple
from copy import deepcopy
from io import StringIO
import itertools
import os
import pickle
import sys
from typing import List, Tuple, Dict

from trec_car import read_data
from trec_car.read_data import (AnnotationsFile, Page, Para,
                                Section, List as ParaList, ParaLink, ParaText, ParaBody)


import tokenizer_cereal


import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, TensorDataset
import tqdm
from keras.preprocessing.sequence import pad_sequences

from .dumps import wrap_open, get_filename_path, is_file


def b2i(number_as_bytes: bytes):
    """
    Convert bytes to ints
    """
    return int.from_bytes(number_as_bytes, "big")

PageFormat = namedtuple("PageFormat", ["id", "text", "link_mentions"])

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
                 cutoff_frequency=0.03,
                 page_lim=-1,
                 token_length=128,
                 clean_cache=False,
                 repreprocess=False,
                 recount=False,
                 ):
        """
        :param cbor_path the relative path of the wikipedia cbor export
        :param partition_path the relative path of the partitions to use
        :param cutoff_frequency the (relative) top-k common entities.
        A value of 1 means all the entities, of 0.5 means the upper half and so on.
        Ties will be broken randomly.
        :param page_lim the number of pages in a partition
        :param token_length return only the first `token_length` tokens of a page.
        :param clean_cache delete the old cache. Implies repreprocess and recount
        :param repreprocess preprocess the text. Implies recount
        :param recount recount the frequencies and update the top-k common entity list.
        """
        self.cbor_path = get_filename_path(cbor_path)
        self.partition_path = get_filename_path(partition_path)
        self.cutoff_frequency = cutoff_frequency
        self.token_length = token_length
        self.cumulated_block_sizes_path = self.partition_path + "/cumulated_block_sizes.npy"

        os.makedirs(self.partition_path, exist_ok=True)
        cache_path = os.path.split(self.cbor_path)[0]
        key_file = os.path.join(cache_path, "sorted_keys.pkl")

        if os.path.isfile(key_file) and not clean_cache:
            with open(key_file, "rb") as pickle_cache:
                self.offsets, self.key_titles, self.key_encoder, \
                    self.valid_keys, self.chosen_freqs = pickle.load(
                        pickle_cache)

            # TODO: should I merge these 2 caches?
            with open(self.cumulated_block_sizes_path, "rb") as fp:
                self.cumulated_block_size = np.load(fp)

            tqdm.tqdm.write("Loaded from cache")
        else:
            tqdm.tqdm.write("Generating key cache")

            self.cbor_toc_annotations = AnnotationsFile(self.cbor_path)
            self.offsets = np.array(
                list(self.cbor_toc_annotations.toc.values()))
            self.offsets.sort()

            self.key_titles = self.extract_readable_key_titles()
            self.key_encoder = dict(zip(self.key_titles, itertools.count()))
            self.key_encoder["PAD"] = 0  # useful for batch transforming stuff

            # preprocess and find the top k unique wikipedia links
            self.valid_keys = set(self.key_encoder.values())

        self.rust_cereal_path = self.partition_path + "/test_rust.cereal"

        if clean_cache or repreprocess:
            self.preprocess(page_lim)

        else:
            try:
                self.tokenizer = tokenizer_cereal.get_default_tokenizer(self.rust_cereal_path)

            # Every rust exception returns a TypeError. Normally get_default_tokenizer
            # may fail if the rust_cereal_path is not existing, but it would be nice to cover
            # other types of exceptions
            # pylint:disable=bare-except
            except:
                tqdm.tqdm.write("Unable to find a tokenizer! Regenerating")
                self.preprocess(page_lim)
            
        if clean_cache or repreprocess or recount:
            freqs = self.count_frequency()
            tqdm.tqdm.write("Obtained link frequency, sorting...")
            freqs_as_pair = list(zip(freqs.values(), freqs.keys()))
            tqdm.tqdm.write("Sorted, filtering the most common links...")

            freqs_as_pair.sort(key=lambda x: -x[0])
            self.chosen_freqs = freqs_as_pair[:int(
                self.cutoff_frequency * len(freqs_as_pair))]
            self.chosen_freqs = set(map(lambda x: x[1], self.chosen_freqs))

            with open(key_file, "wb") as pickle_cache:
                pickle.dump((self.offsets, self.key_titles, self.key_encoder,
                             self.valid_keys, self.chosen_freqs), pickle_cache)

            tqdm.tqdm.write("Cache was generated")

        # map the original "sparser" keys to a smaller set - as long as token_length in theory

        self.max_entity_num = len(self.chosen_freqs)
        self.key_restrictor = dict(
            zip(self.chosen_freqs, range(self.max_entity_num)))

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

            raise Exception("Wrong header")

        key_titles = set()

        # If reloaded for a second time, this should be way faster.
        #with mmap.mmap(self.cbor_toc_annotations.cbor.fileno(), 0, mmap.MAP_PRIVATE) as cbor_file:
        cbor_file = self.cbor_toc_annotations.cbor
        for offset in tqdm.tqdm(self.offsets, desc="Extracting human-readable page titles"):
            key_titles.add(extract_from_key(offset))

        return key_titles

    def __len__(self):
        return self.cumulated_block_size[-1]

    @staticmethod
    def get_attention_mask(tokens: List[int]):
        """
        Get an attention mask. Padding tokens are automatically masked out.
        :param tokens the tokens to process
        """

        return [1 if tok != 0 else 0 for tok in tokens]
    
    @staticmethod
    def get_boundaries_masked(tokens: List[int], entities: List[int]):
        """
        Get a boundary list and a partially masked entity output label. 20%
        of randomly chosen entity mentions are removed.
        
        :param tokens the output labels from the tokenizer
        :returns a pair (output_tokkens, output_entities, output_bio), both with the same shapes of the input.
        """
        
        last_seen = 0
        
        spans = []
        
        output_bio = [0] * len(tokens)
        output_tokens = deepcopy(tokens)
        output_entities = deepcopy(entities)
        
        # extend to cover the edge case of boundaries going till the end
        for idx, tok in enumerate(entities + [0]):
            if tok != 0:
                if last_seen != tok:
                    spans.append([idx, 0])
                if last_seen != 0:
                    spans[-1][1] = idx
                    

            elif last_seen != 0:
                spans[-1][1] = idx
            last_seen = tok
            
        # "We remove 20% of randomly chosen entity mentions"
        spans_to_take = np.random.choice([False, True], len(spans), p=(.2, .8))

        for span, to_take in zip(spans, spans_to_take):
            if to_take:
                output_bio[span[0]] = 1
                for i in range(span[0]+1, span[1]):
                    output_bio[i] = 2
                
            else:
                for i in range(span[0], span[1]):
                    output_tokens[i] = 103 # [MASK]
                    output_entities[i] = 103 # [MASK]
        return output_tokens, output_entities, output_bio
        

    def preprocess_page(self, enumerated_page: Tuple[int, Page]) -> PageFormat:
        """
        Transform a list of pages into a flattened representation that can
        then be easily (de)serialized.
        """

        id, page = enumerated_page

        # For the sake of easy link spans they are byte oriented to make
        # it easier for the rust std
        page_content = StringIO()
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
            page_content.write(body.get_text())

        def handle_paralink(body: ParaLink):
            encoded_link = encode_link(body.page)
            start_byte_span = page_content.tell()
            end_byte_span = start_byte_span + len(body.get_text()) - 1
            page_content.write(body.get_text())

            links.append((encoded_link, start_byte_span, end_byte_span))

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
        
        return PageFormat(id, page_content.getvalue(), links)

    def preprocess(self, limit: int):
        """
        Transform the CBOR file into a rust-cerealised version that is already tokenized
        and ready for use.

        :param limit process the first `limit` pages.
        """

        # save cumulated block sizes
        with open(self.cbor_path, "rb") as cbor_fp:
            if getattr(self, "tokenizer", None) is not None:
                del self.tokenizer # ensure the destructor is called (?)

            self.tokenizer = tokenizer_cereal.TokenizerCereal(self.rust_cereal_path,
                map(self.preprocess_page, enumerate(read_data.iter_annotations(cbor_fp))),
                limit)

        blocks_per_page = [int(np.floor(length / self.token_length))
            for length in self.tokenizer.article_lengths if length > 0]

        with open(self.cumulated_block_sizes_path, "wb") as fp:
            self.cumulated_block_size = np.cumsum(blocks_per_page)
            np.save(fp, self.cumulated_block_size)

    def count_frequency(self) -> Dict[int, int]:
        """
        Get the frequency count. Just a mere wrapper over the Rust-made module.
        """
        return self.tokenizer.get_frequency_count()

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
        """
        Return a tensor for the given index.

        The first row is the token embedding via WordPiece ids.
        The second row is the BERT attention mask.
        The third row is the expected Entity Linking output.

        Given that we work with token blocks, we convert the provided indices
        to the correct page and block offsets.

        :param idx the index of the block to fetch.
        """
        if not isinstance(idx, int):
            idx = idx[0]

        cumulated = self.cumulated_block_size
        page_idx = bisect.bisect_right(cumulated, idx)
        page_block_nums = cumulated[page_idx] - \
            (cumulated[page_idx-1] if page_idx > 0 else 0)
        block_offset = idx - (cumulated[page_idx-1] if page_idx > 0 else 0)

        # Too tired to properly handle this edge case
        if page_block_nums == 0:
            page_idx = 0
            block_offset = 0

        toks, links = self.tokenizer.get_slice(page_idx, block_offset, self.token_length)

        links = [self.key_restrictor.get(x, 0) for x in links]
        
        masked_toks, masked_links, masked_bio = self.get_boundaries_masked(toks, links)
        
        toks, links, masked_toks, masked_links, masked_bio = [
            torch.LongTensor(pad_sequences([x], maxlen=self.token_length,
                                dtype="long", value=0.0,
                                truncating="post", padding="post")[0]) \
                    .squeeze() for x in (toks, links, masked_toks,
                                            masked_links, masked_bio)
        ]
        
        # attend everywhere it is not padded
        attns = torch.where(toks != 0, 1, 0)

        return (masked_toks, toks, masked_links, links, masked_bio, attns)


class BIO:
    """
    Load and process the GBK corpus
    """

    def __init__(self, dataset_dir, token_length, clean_cache=False):
        """
        :param clean_cache if True clean the cache and restart tokenization again
        """
        self.token_length = token_length
        cache_path = dataset_dir + ".pkl"
        
        if clean_cache or not is_file(cache_path):
            # Download tokenizer from S3 and cache.
            self.tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                                            'tokenizer', 'bert-base-uncased')
            data_pd = self.__load_dataset(dataset_dir)
            self.__simplify_bio(data_pd)
            utterances, labels = self.__aggregate(data_pd)

            self.bio_values = list(set(data_pd["bio"].values))
            self.bio_values.append("PAD")

            # BIO tag to a numerical index (yes, this is dumb way to make an enum)
            # Apparently one row is misclassified as p?
            self.bio2idx = {t: i for i, t in enumerate(self.bio_values)}
            self.bio2idx['p'] = self.bio2idx['O']

            # Tokenize and split
            tokenized_texts_labels = [
                self.__tokenize_preserve_labels(sent, labs) for sent, labs in
                zip(utterances, labels)
            ]

            tokenized_texts = [token_label_pair[0]
                               for token_label_pair in tokenized_texts_labels]
            tokenized_labels = [token_label_pair[1]
                                for token_label_pair in tokenized_texts_labels]

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

            self.attention_mask = [[float(i != 0.0) for i in ii]
                                   for ii in self.input_ids]

            # Tensorize...
            # pylint:disable=not-callable
            self.input_ids = torch.tensor(
                self.input_ids)
            self.labels = torch.tensor(self.labels)  # pylint:disable=not-callable
            # pylint:disable=not-callable
            self.attention_mask = torch.tensor(
                self.attention_mask)
            
            with wrap_open(cache_path, "wb") as pickle_cache:
                pickle.dump((self.bio_values, self.bio2idx, self.input_ids, self.labels, self.attention_mask), pickle_cache)

            tqdm.tqdm.write("Cache was generated")

        else:
            with wrap_open(cache_path, "rb") as pickle_cache:
                self.bio_values, self.bio2idx, self.input_ids, self.labels, self.attention_mask = pickle.load(pickle_cache)
                
            tqdm.tqdm.write("Loaded from cache")

    def __load_dataset(self, dataset_path):
        """
        Load the dataset

        :param dataset_dir the relative path where the dataset is.
        """

        names = []

        with wrap_open(dataset_path, "r", encoding="latin1") as f:
            names = ["index"] + f.readline().strip().split(",")[1:]
            names = names + list(range(34 - len(names)))

        with wrap_open(dataset_path, "rb") as f:
            f.readline()  # skip the first line
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
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def get_pytorch_dataset(self) -> Dataset:
        """
        Return the whole dataset as a TensorDataset. 
        """
        return TensorDataset(self.input_ids, self.attention_mask, self.labels)
