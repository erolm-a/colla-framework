'''
Convenience tools for parsing the Wikipedia CBOR structure and a number of evaluation datasets.
'''
import bisect
from collections import defaultdict, namedtuple
from copy import deepcopy
from io import StringIO
import itertools
import math
import os
import pickle
from typing import List, Tuple, Dict, Sequence

from trec_car import read_data
from trec_car.read_data import (AnnotationsFile, Page, Para,
                                Section, List as ParaList, ParaLink, ParaText, ParaBody)


import tokenizer_cereal


import numpy as np

import datasets
import torch
from torch.utils.data import Dataset
import tqdm
from keras.preprocessing.sequence import pad_sequences

from .dumps import get_filename_path
from .vocabs import load_tokenizer


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
                 token_length=512,
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
        self.tokenizer = None

        os.makedirs(self.partition_path, exist_ok=True)
        cache_path = os.path.split(self.cbor_path)[0]
        key_file = os.path.join(cache_path, "sorted_keys.pkl")

        if os.path.isfile(key_file) and not clean_cache:
            with open(key_file, "rb") as pickle_cache:
                self.offsets, self.key_titles, self.key_encoder, \
                    self.valid_keys, self.chosen_freqs, self.blocks_per_page = pickle.load(
                        pickle_cache)

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

        if clean_cache or repreprocess or not os.path.exists(self.rust_cereal_path):
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
                             self.valid_keys, self.chosen_freqs, self.blocks_per_page), pickle_cache)

            tqdm.tqdm.write("Cache was generated")

        # map the original "sparser" keys to a smaller set - as long as token_length in theory

        self.max_entity_num = len(self.chosen_freqs)
        self.key_restrictor = dict(
            zip(self.chosen_freqs, range(self.max_entity_num)))

        self.key_decoder = dict([(b, a) for a, b in self.key_restrictor.items()])
        self.key_titles_vec = self.extract_toc_titles()

        # == ITERATION CONTROL ==
        self.last_page = 0
        self.last_block = 0
        self.start_page_idx = 0
        self.end_page_idx = len(self.blocks_per_page)
        self.cur_block = 0

        # length
        self.cumulated_block_size = np.cumsum(self.blocks_per_page)
        self.length = self.cumulated_block_size[-1]


    def __len__(self):
        return self.length
    
    def extract_toc_titles(self) -> List[str]:
        """
        Extract a list of titles from the ToC.
        """

        return list(self.key_encoder.keys())


    def extract_readable_key_titles(self):
        """
        Build a set of human-readable names of CBOR entries.
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
                    output_tokens[i] = 103  # [MASK]
                    output_entities[i] = 103  # [MASK]
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

        with open(self.cbor_path, "rb") as cbor_fp:
            # TODO: revert to a function-based approach
            tokenizer = tokenizer_cereal.TokenizerCereal(self.rust_cereal_path,
                                                         map(self.preprocess_page, enumerate(
                                                             read_data.iter_annotations(cbor_fp))),
                                                         limit)

        self.blocks_per_page = [int(np.ceil(length / self.token_length))
                                for length in tokenizer.article_lengths if length > 0]

    def count_frequency(self) -> Dict[int, int]:
        """
        Get the frequency count. Just a mere wrapper over the Rust-made module.
        """

        tokenizer = tokenizer_cereal.get_default_tokenizer(
            self.rust_cereal_path)
        return tokenizer.get_frequency_count()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # for multithreading be sure to recalculate the start and end iterator index
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = int(math.ceil(self.end_page_idx / float(num_workers)))
            self.start_page_idx = per_worker * worker_id
            self.end_page_idx = min(
                self.start_page_idx + per_worker, len(self.blocks_per_page))

        return iter(self._getitem())

    def convert_idx_to_page_block(self, idx: int) -> Tuple[int, int]:
        cumulated = self.cumulated_block_size
        page_idx = bisect.bisect_right(cumulated, idx)
        page_block_nums = cumulated[page_idx] - \
            (cumulated[page_idx-1] if page_idx > 0 else 0)
        block_offset = idx - (cumulated[page_idx-1] if page_idx > 0 else 0)

        if page_block_nums == 0:
            page_idx = 0
            block_offset = 0 

        return page_idx, block_offset

    def process_tokenizer_output(self, toks: List[int], links: List[int]
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor, torch.Tensor]:
        links = [self.key_restrictor.get(x, 0) for x in links]

        masked_toks, masked_links, masked_bio = self.get_boundaries_masked(
            toks, links)

        toks_tensor, links_tensor, masked_toks_tensor, masked_links_tensor, masked_bio_tensor = [
            torch.LongTensor(pad_sequences([x], maxlen=self.token_length,
                                           dtype="long", value=0.0,
                                           truncating="post", padding="post")[0])
            .squeeze() for x in (toks, links, masked_toks,
                                 masked_links, masked_bio)
        ]

        # attend everywhere it is not padded
        attns = torch.where(toks_tensor != 0, 1, 0)

        return (masked_toks_tensor, toks_tensor, masked_links_tensor,
                links_tensor, masked_bio_tensor, attns)

    def _getitem(self):
        """
        Return a tensor for the given index.

        The first row is the token embedding via WordPiece ids.
        The second row is the BERT attention mask.
        The third row is the expected Entity Linking output.

        Given that we work with token blocks, we convert the provided indices
        to the correct page and block offsets.

        :param idx the index of the block to fetch.
        """

        if not self.tokenizer:
            self.tokenizer = tokenizer_cereal.get_default_tokenizer(
                self.rust_cereal_path)

        while self.start_page_idx < self.end_page_idx:
            page_lim = self.blocks_per_page[self.start_page_idx]

            if self.cur_block == page_lim:
                self.start_page_idx += 1
                self.cur_block = 0

            if self.cur_block == 0:
                self.last_page = self.tokenizer.get_next_slice()

            toks = self.last_page[0][self.cur_block *
                                     self.token_length: (self.cur_block + 1)*self.token_length]
            links = self.last_page[1][self.cur_block *
                                      self.token_length: (self.cur_block + 1)*self.token_length]
            

            masked_toks, toks, masked_links, links, masked_bio, attns = \
                self.process_tokenizer_output(toks, links)

            self.cur_block += 1

            yield (masked_toks, toks, masked_links, links, masked_bio, attns)
    
    def __getitem__(self, idx):
        """
        Provide an interface for a map-style, O(log(N)) approach for accessing a random block
        """

        if not self.tokenizer:
            self.tokenizer = tokenizer_cereal.get_default_tokenizer(
                self.rust_cereal_path)

        page_idx, block_offset = self.convert_idx_to_page_block(idx)

        toks, links = self.tokenizer.get_slice(page_idx, block_offset,
                            self.token_length)
        
        return self.process_tokenizer_output(toks, links)

    def decode_compressed_entity_ids(self, entity_batches: Sequence[Sequence[int]]) -> List[List[str]]:
        return [[self.key_titles_vec[self.key_decoder.get(int(idx), 0)] for idx in batch] for batch in entity_batches]

class SQuADDataloader():
    """
    This is a convenience class for accessing the SQuAD dataset as a Pytorch dataloader
    """

    # Fancy getitem to work with samplers

    class SquadDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            if type(idx) == list or type(idx) == torch.tensor:
                return [self.dataset[i] for i in idx]
            else:
                # TODO: this is very fragile
                if type(idx) == np.int64:
                    idx = idx.item()
                return self.dataset[idx]

    def __init__(self, block_size=512):
        """
        Set up a Squad dataloader pipeline.

        :param block_size The model's block size. We drop questions that do not fit the model.
        """
        self.tokenizer = load_tokenizer('bert-base-uncased')
        self.dataset = datasets.load_dataset("squad")

        def encode(examples):
            contexts = examples['context']
            questions = examples['question']

            encoded_full_sentences = [self.tokenizer.encode(context, question)
                                      for context, question in zip(contexts, questions)]

            for sentence in encoded_full_sentences:
                sentence.pad(block_size)
                sentence.truncate(block_size)

            answers = examples['answers']
            # TODO: tag unanswerable questions with -1
            answers_start = [answer['answer_start'][0]
                             for answer in answers]  # this is a byte offset

            answers_end = [answer_start + len(answer['text'][0]) for answer, answer_start in
                           zip(answers, answers_start)]

            answer_start_idx = [-1] * len(answers_start)
            answer_end_idx = [-1] * len(answers_end)

            for answer_idx, (encoded_sentence, answer_start, answer_end) in \
                    enumerate(zip(encoded_full_sentences, answers_start, answers_end)):
                for idx, (start_offset, end_offset) in enumerate(encoded_sentence.offsets):
                    # Separation tokens have 0-len span.
                    if start_offset == end_offset:
                        continue

                    if start_offset >= answer_start and answer_start_idx[answer_idx] == -1:
                        answer_start_idx[answer_idx] = idx

                    if end_offset >= answer_end and answer_end_idx[answer_idx] == -1:
                        answer_end_idx[answer_idx] = idx
                        break

                    # assign answer_end_idx to the SEP position if nothing is found
                    if encoded_sentence.token_to_chars(idx) == "[SEP]":
                        answer_end_idx[answer_idx] = idx - 1
                        break

            input_ids = [x.ids for x in encoded_full_sentences]
            type_ids = [x.type_ids for x in encoded_full_sentences]
            attention_mask = [x.attention_mask for x in encoded_full_sentences]

            return {'input_ids': input_ids, 'attention_mask': attention_mask,
                    'token_type_ids': type_ids, 'answer_start': answer_start_idx,
                    'answer_end': answer_end_idx}

        self.tokenized_dataset = self.dataset.map(encode, batched=True)
        #self.tokenized_dataset.set_format(type="torch", columns=['input_ids', 'token_type_ids',
        #                                                         'attention_mask', 'answer_start',
        #                                                         'answer_end'])

    @property
    def train_dataset(self):
        return SQuADDataloader.SquadDataset(self.tokenized_dataset["train"])

    @property
    def validation_dataset(self):
        return SQuADDataloader.SquadDataset(self.tokenized_dataset["validation"])

    def reconstruct_sentences(
                             self,
                             input_ids_list: List[List[int]],
                             answers_start: List[int],
                             answers_end: List[int]
                             ) -> List[str]:
        """
        Reconstruct the sentences given a list of token ids and a span.
        Unfortunately there is no way to do that efficiently given that spans are ragged.

        :param input_ids_list
        :param answers_start
        :param answers_end

        :returns a list of strings
        """

        answers = [input_ids[answer_start:answer_end+1] for input_ids, answer_start, answer_end 
            in zip(input_ids_list, answers_start, answers_end)]

        return self.tokenizer.decode_batch(answers)
