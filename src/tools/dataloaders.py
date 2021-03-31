'''
Convenience tools for parsing the Wikipedia CBOR structure and a number of evaluation datasets.
'''
import bisect
from collections import defaultdict, namedtuple
from copy import deepcopy
from io import StringIO
import re

import itertools
import math
import os
import pickle
from heapq import merge
from typing import List, Tuple, Dict, Sequence, Optional, NamedTuple, NewType

from trec_car import read_data
from trec_car.read_data import (AnnotationsFile, Page, Para,
                                Section, List as ParaList, ParaLink, ParaText, ParaBody)


import tokenizer_cereal


import numpy as np

import datasets
import pprint
import pandas as pd
from transformers import AutoTokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Whitespace
from transformers.data.processors import squad
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

from tools.dumps import get_filename_path
from tools.strings import MyRecordTrie


def b2i(number_as_bytes: bytes):
    """
    Convert bytes to ints
    """
    return int.from_bytes(number_as_bytes, "big")


# Token and token lists returned from a Whitespace tokenization
Token = NewType("Token", Tuple[str, Tuple[int, int]])
TokenizedText = NewType("TokenizedText", List[Token])
Link = NewType("Link", Tuple[int, int, int])


class PageFormat(NamedTuple):
    id: int
    title: str
    #text: str
    pretokenized_text: TokenizedText
    link_mentions: List[Link]


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

    def __init__(
        self,
        cbor_path: str,
        partition_path: str,
        cutoff_frequency=0.03,
        page_lim=-1,
        token_length=512,
        clean_cache=False,
        repreprocess=False,
    ):
        """
        :param cbor_path the relative path of the wikipedia cbor export
        :param partition_path the relative path of the partitions to use
        :param cutoff_frequency the (relative) top-k common entities.
        A value of 1 means all the entities, of 0.5 means the upper half and so on.
        Ties will be broken randomly.
        :param page_lim the number of pages in a partition
        :param token_length return only the first `token_length` tokens of a page.
        :param clean_cache delete the old cache. Implies repreprocess
        :param repreprocess preprocess the text.
        """
        self.cbor_path = get_filename_path(cbor_path)
        self.partition_path = get_filename_path(partition_path)
        self.cutoff_frequency = cutoff_frequency
        self.token_length = token_length
        self.tokenizer = None

        os.makedirs(self.partition_path, exist_ok=True)
        cache_path = os.path.split(self.cbor_path)[0]

        cache_name = f"titles.pkl"
        key_file = os.path.join(cache_path, cache_name)

        if os.path.isfile(key_file) and not clean_cache:
            with open(key_file, "rb") as pickle_cache:
                self.key_titles = pickle.load(pickle_cache)

            tqdm.write("Loaded from cache")
        else:
            tqdm.write("Generating article title cache")

            self.cbor_toc_annotations = AnnotationsFile(self.cbor_path)
            offsets = np.array(
                list(self.cbor_toc_annotations.toc.values()))
            offsets.sort()

            self.key_titles = self.extract_readable_key_titles_new(offsets)

            with open(key_file, "wb") as pickle_cache:
                pickle.dump(self.key_titles, pickle_cache)

            tqdm.write("Cache was generated")

        if page_lim < 0:
            page_lim = len(self.key_titles)

        self.key_encoder = dict(zip(self.key_titles, itertools.count()))

        self.rust_cereal_path = self.partition_path + \
            f"/tokenized_{page_lim}.cereal"

        if clean_cache or repreprocess or not os.path.exists(self.rust_cereal_path):
            tqdm.write("Generating cereal-ised cache")
            self.preprocess(page_lim)

        # TODO: should this be in a separate cache?
        freqs = self.count_frequency()
        tqdm.write("Obtained link frequency, sorting...")
        freqs_as_pair = list(zip(freqs.values(), freqs.keys()))
        tqdm.write("Filtering by most common links...")

        freqs_as_pair.sort(key=lambda x: -x[0])
        self.chosen_freqs = freqs_as_pair[:int(
            self.cutoff_frequency * len(freqs_as_pair))]
        self.chosen_freqs = set(map(lambda x: x[1], self.chosen_freqs))
        # map the original "sparser" keys to a smaller set - as long as token_length in theory

        self.max_entity_num = len(self.chosen_freqs)
        self.key_restrictor = dict(
            zip(self.chosen_freqs, range(self.max_entity_num)))

        tqdm.write("Building key decoder...")
        self.key_decoder = {b: a for a, b in self.key_restrictor.items()}

        # == LENGTH ==
        # No, tokenizer MUST NOT be a member here due to multiprocessing issues
        # (file descriptors get shared across forks, thusing causing internal crashes)
        tokenizer = tokenizer_cereal.get_default_tokenizer(
            self.rust_cereal_path)

        self.blocks_per_page = [int(np.ceil(length / self.token_length))
                                for length in tokenizer.article_lengths if length > 0]

        # == ITERATION CONTROL ==
        self.last_page = 0
        self.last_block = 0
        self.start_page_idx = 0
        self.end_page_idx = len(self.blocks_per_page)
        self.cur_block = 0

        self.cumulated_block_size = np.cumsum(self.blocks_per_page)
        self.length = self.cumulated_block_size[-1]

        # UTILS
        self.normalizer = BertNormalizer()
        self.splitter = Whitespace()

    def __len__(self):
        return self.length

    def extract_toc_titles(self) -> List[str]:
        """
        Extract a list of titles from the ToC.
        """

        return list(self.key_encoder.keys())

    def extract_readable_key_titles(self, offsets: List[int]):
        """
        Build a list of human-readable names of CBOR entries.
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

        key_titles = ["PAD"]

        # If reloaded for a second time, this should be way faster.
        #with mmap.mmap(self.cbor_toc_annotations.cbor.fileno(), 0, mmap.MAP_PRIVATE) as cbor_file:
        cbor_file = self.cbor_toc_annotations.cbor
        for offset in tqdm(offsets, desc="Extracting human-readable page titles"):
            key_titles.append(extract_from_key(offset))

        return key_titles

    def extract_readable_key_titles_new(
        self,
        offsets: List[int]
    ) -> List[str]:
        # Yes, this is a slow and heavily inefficient implementation.
        # But it is also true we perform this only once

        key_titles = ["PAD"]
        with open(self.cbor_path, "rb") as f:
            for page in tqdm(read_data.iter_annotations(f), total=len(offsets), desc="Extract human-readable article titles"):
                if isinstance(page.page_type, read_data.ArticlePage):
                    key_titles.append(page.page_name)
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
            output_bio[span[0]] = 1
            for i in range(span[0]+1, span[1]):
                output_bio[i] = 2

            if not to_take:
                for i in range(span[0], span[1]):
                    output_tokens[i] = 103  # [MASK]
                    # TODO allocate a special [MASK] token for links.
                    output_entities[i] = 0
        return output_tokens, output_entities, output_bio

    def preprocess_page(
        self,
        enumerated_page: Tuple[int, Page]
    ):
        """
        Transform a list of pages into a flattened representation that can
        then be easily (de)serialized.
        """

        id, page = enumerated_page

        # For the sake of easy link spans they are byte oriented to make
        # it easier for the rust std
        # page_content = StringIO()
        split_content = []
        orig_page_content = StringIO()
        prev_page_length = 0
        prev_body = ""
        links = []

        # Encode a link. Cast to padding if the link was not "common".
        # Call this method only after preprocessing has been done!
        def encode_link(link):
            #return self.key_encoder.get(link, 0)
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
            nonlocal prev_page_length
            nonlocal prev_body
            cur_body = body.get_text()

            split_body = self.splitter.pre_tokenize_str(
                self.normalizer.normalize_str(cur_body))
            #print('from handle_paratext: "' + body.get_text() + '"')

            # take care of the space...
            running_prefix = 0
            current_page_length = prev_page_length + len(cur_body)
            if len(split_content) > 0:
                running_prefix = prev_page_length  # split_content[-1][1][1]

            orig_page_content.write(cur_body)

            #print(f"After skipping: {orig_page_content.getvalue()[running_prefix:]}")

            split_body = [(text, (begin_offset + running_prefix,
                                  end_offset + running_prefix)) for text, (begin_offset, end_offset) in split_body]

            # print(split_body)

            split_content.extend(split_body)

            prev_page_length = current_page_length
            prev_body = body.get_text()

        def handle_paralink(body: ParaLink):
            nonlocal prev_body
            nonlocal prev_page_length
            encoded_link = encode_link(body.page)
            cur_body = body.get_text()

            split_body = self.splitter.pre_tokenize_str(
                self.normalizer.normalize_str(cur_body))
            #print('from handle_paralink: "' + body.get_text() + '"')

            running_prefix = 0
            current_page_length = prev_page_length + len(cur_body)
            if len(split_content) > 0:
                running_prefix = prev_page_length

            orig_page_content.write(cur_body)

            split_body = [(text,
                (begin_offset + running_prefix, end_offset + running_prefix)) \
                    for text, (begin_offset, end_offset) in split_body]

            split_content.extend(split_body)

            if len(split_body) > 0:
                # end_byte_span = split_body[-1][1][1] - 1
                start_mention_idx = len(split_content) - len(split_body)
                links.append(
                    (encoded_link, start_mention_idx, len(split_content)))

            #for tok, (begin, end) in split_body:
            #    assert tok == orig_page_content.getvalue()[begin:end], f"generated {orig_page_content.getvalue()[begin:end]} but expected {tok}"

            prev_page_length = current_page_length
            prev_body = body.get_text()

        def nothing():
            return lambda body: None

        handler = defaultdict(nothing, {Section: handle_section,
                                        Para: handle_para,
                                        ParaList: handle_list,
                                        ParaLink: handle_paralink,
                                        ParaText: handle_paratext})

        def visit_section(skel):
            # Recur on the sections
            handler[type(skel)](skel)

        for skel in page.skeleton:
            visit_section(skel)

        return orig_page_content.getvalue(), PageFormat(id, page.page_name, split_content, links)

    def autolink(
        self,
        page_id: int,
        title: str,
        text: str,
        tokenized_text: TokenizedText,
        links: List[Link]
    ) -> List[Link]:
        exact_mentions = {}

        # TODO: deal with ambiguities...
        exact_mentions[title] = self.normalizer.normalize_str(page_id)

        remapped_links = []
        for link in links:
            # revert the tokenization algorithm
            start_byte = tokenized_text[link[1]][1][0]
            end_byte = tokenized_text[link[2]-1][1][1]

            exact_mentions[self.normalizer.normalize_str(
                text[start_byte:end_byte])] = link[0]
            remapped_links.append((link[0], start_byte, end_byte))

        trie = MyRecordTrie(
            map(lambda x: (x[0], (x[1],)), exact_mentions.items()))
        patterns = sorted(trie.search_longest_patterns(
            tokenized_text), key=lambda x: x[1])  # sort by apparition
        #patterns_across = list(sorted(self.all_pages_references.search_longest_patterns(
        #    tokenized_text), key=lambda x: x[1]))
        #merged_patterns = list(
        #    merge(patterns, patterns_across, key=lambda x: x[1]))

        link = None
        link_idx = 0
        new_links = []

        for title, idx, new_link_id, _ in patterns:

            new_link = (new_link_id, idx, idx + len(title))
            # print(title, new_link)

            if link_idx < len(remapped_links):
                link = remapped_links[link_idx]

            while link_idx < len(remapped_links) - 1 and idx > link[2]:
                link_idx += 1
                link = remapped_links[link_idx]
                #print("Increasing")

            if link_idx >= len(remapped_links):
                link = None

            if link is None or idx < link[1]:
                new_links.append(new_link)

        return list(merge(remapped_links, new_links, key=lambda x: x[1]))

    def preprocessing_pipeline(
        self,
        enumerated_page: Tuple[int, Page]
    ) -> PageFormat:
        """
        Pipeline for preprocessing.
        """

        text, page_output = self.preprocess_page(enumerated_page)
        new_links = self.autolink(
            page_output.id, page_output.title,
            text, page_output.pretokenized_text,
            page_output.link_mentions)

        return PageFormat(page_output.id, page_output.title, page_output.pretokenized_text, new_links)

    def preprocess(self, limit: int):
        """
        Transform the CBOR file into a rust-cerealised version that is already tokenized
        and ready for use.

        :param limit process the first `limit` pages.
        """

        def _preprocess():
            #self.all_pages_references = MyRecordTrie(map(
            #    lambda x: (self.normalizer.normalize_str(x[1]), (x[0],)),
            #    enumerate(self.key_titles)))

            with open(self.cbor_path, "rb") as cbor_fp:
                counter = 1  # 0 is for PAD

                annotator_iter = read_data.iter_annotations(cbor_fp)
                while counter <= limit:
                    try:
                        page = next(annotator_iter)
                        if not isinstance(page.page_type, read_data.ArticlePage):
                            continue
                        yield self.preprocessing_pipeline((counter, page))
                        counter += 1
                    except StopIteration:
                        break

        tokenizer_cereal.TokenizerCereal(
            self.rust_cereal_path, _preprocess(), limit)

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
        return [[self.key_titles[self.key_decoder.get(int(idx), 0)] for idx in batch] for batch in entity_batches]


class SQuADDataloader():
    """
    This is a convenience class for accessing the SQuAD dataset as a Pytorch dataloader
    """

    # Fancy getitem to work with samplers

    class SquadDataset(Dataset):
        def __init__(
            self,
            dataset: Dataset,
            custom_len: Optional[int] = None
        ):
            self.dataset = dataset
            self.length = custom_len if custom_len else len(self.dataset)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            if type(idx) == list or type(idx) == torch.tensor:
                return [self.dataset[i] for i in idx]
            else:
                # FIXME: this is very fragile
                if type(idx) == np.int64:
                    idx = idx.item()
                return self.dataset[idx]

    def __init__(
        self,
        max_seq_length=512,
        max_query_length=384,
        doc_stride=128,
    ):
        """
        Set up a Squad dataloader pipeline.
        """

        # TODO: Use Transformer's interface for the tokenizer rather than the crude one
        # TODO: Deprecate vocabs.py
        # self.tokenizer = load_tokenizer('bert-base-uncased')

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=False)
        self.dataset = datasets.load_dataset("squad")

        squad.squad_convert_example_to_features_init(self.tokenizer)

        def encode(examples):
            qas_id = examples["id"]
            question_text = examples["question"]
            answers = examples["answers"]
            answer_text, start_position_characters = zip(*[
                (answer["text"][0], answer["answer_start"][0]) for answer in answers])
            answer_text = list(answer_text)
            start_position_characters = list(start_position_characters)
            titles = examples["title"]
            context_text = examples["context"]

            # Avoid reshuffling issues by performing a late join
            examples_pd = pd.DataFrame({
                "id": qas_id,
                "answers": answers,
                "title": titles,
                "context": context_text,
                "question": question_text
            })

            squad_examples = list(squad.SquadExample(*args) for args in zip(
                qas_id,
                question_text,
                context_text,
                answer_text,
                start_position_characters,
                titles,
                answers
            ))

            features = squad.squad_convert_examples_to_features(
                squad_examples,
                self.tokenizer,
                max_seq_length,
                doc_stride,
                max_query_length,
                True,
            )

            # In case of split context create singleton lists.
            result = pd.DataFrame([vars(feature) for feature in features])
            result_dropped = result.drop(["example_index", "token_is_max_context",
                                          "encoding", "token_to_orig_map", "cls_index", "paragraph_len",
                                          "unique_id"], axis=1)

            # the HF processor silently removes problematic rows. In this case we have no choice but
            # to add singletons for now and filter them out later

            result_grouped = result_dropped.groupby(
                "qas_id", as_index=False).agg(list)

            result_grouped = examples_pd.set_index("id").join(
                result_grouped.set_index("qas_id"),
                how="left"
            ).reset_index()

            result_as_dict = result_grouped.to_dict()

            result = {}

            for k, v in result_as_dict.items():
                values = list(v.values())
                for i in range(len(values)):
                    if values[i] != values[i]:  # NaN:
                        values[i] = []
                result[k] = values

            return result

        self.tokenized_dataset = self.dataset.map(encode, batched=True) \
            .filter(lambda example: len(example["input_ids"]) > 0)

        # Ready-to-use datasets, compatible with samplers if needed.
        self.train_dataset = SQuADDataloader.SquadDataset(
            self.tokenized_dataset["train"]
        )
        self.dev_train_dataset = SQuADDataloader.SquadDataset(
            self.tokenized_dataset["train"], int(len(self.train_dataset)*0.01)
        )
        self.validation_dataset = SQuADDataloader.SquadDataset(
            self.tokenized_dataset["validation"]
        )

        self.dev_validation_dataset = SQuADDataloader.SquadDataset(
            self.tokenized_dataset["validation"],
            int(len(self.validation_dataset))
        )

        # We can't export to pytorch's tensors because we also need the answers sublist!
        #self.tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'token_type_ids',
        #                                                         'answer_start', 'answer_end'])
