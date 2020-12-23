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
import itertools
import mmap
import os
import pickle
import random
import subprocess

from typing import List, Union, Iterable, Tuple

def partition(cbor_dump_path: str, destination: str, num_partitions=100, calculate_only=False):
    """
    Partition a given CBOR file with a toc to a number of given partitions.

    :param toc the position of the cbor dump
    :param destination the destination of the partitions
    :param calculate_only if True do not actually make the partitions.
    """

    with wrap_open(cbor_dump_path + ".toc", "rb") as f:
        toc_file = cbor.load(f)
    
    offsets = list(toc_file.values())
    offsets.sort()
    
    destination = get_filename_path(destination)
    os.makedirs(destination, exist_ok=True)

    # bucketize
    partition_mapping_fname = f"{destination}/partition_offsets.txt"

    partition_mapping_file = wrap_open(partition_mapping_fname, "w")

    block_size = len(offsets) // num_partitions

    for num_partition in range(num_partitions):
        i = num_partition * block_size
        j = min (i + block_size, len(offsets) - 1)

        print(i, j)

        start_offset, end_offset = offsets[i], offsets[j]

        count = end_offset - start_offset
        print(count)

        partition_fname = f"{destination}/{num_partition}.cbor"
        partition_toc_fname = f"{destination}/{num_partition}.cbor.toc"
        # partition_size = end_offset - start_offset
        
        if not calculate_only:
            # Invoke dd and do the actual splitting
            # skip is defined in terms of the input size
            # keep obs high to ensure decent throughput.
            # use tqdm to get a nice status bar for dd (will it work?)
            # I am aware os.system is an antipattern.
            # I am so tired right now and Jeff is gonna kill me.
            os.system(" ".join(["dd", f"if={get_filename_path(cbor_dump_path)}",
                f"of={partition_fname}",
                "ibs=1", "obs=1M", f"skip={start_offset}",
                f"count={end_offset-start_offset}",
                "status=progress"]))
            
            # tqdm.tqdm.write(proc.stdout.readline().decode('utf-8'))
        
        # write the per-partition toc file
        with wrap_open(partition_toc_fname, "w") as output_toc:
            for off in offsets[i:j+1]:
                output_toc.write(str(off - start_offset) + "\n")
        
        partition_mapping_file.write(f"{start_offset}\n")
    
    partition_mapping_file.close()


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
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers',
                                        'tokenizer', 'bert-base-uncased')
        
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
            # self.key_decoder = dict(zip(self.key_encoder.keys(), self.key_encoder.values()))

            # preprocess and find the top k unique wikipedia links
            self.total_freqs = self.extract_links(num_partitions, partition_page_lim=page_lim)
            threshold_value = torch.kthvalue(self.total_freqs,
                            len(self.total_freqs) - (max_entity_num - 1))[0]
            self.valid_keys = set(torch.nonzero(
                self.total_freqs >= threshold_value).squeeze().tolist())
            
            with open(key_file, "wb") as pickle_cache:
                pickle.dump((self.offsets, self.key_titles, self.key_encoder,
                             self.valid_keys), pickle_cache)

        # map the original "sparser" keys to a smaller set - as long as token_length in theory
        self.key_restrictor = dict(zip(self.valid_keys, range(self.max_entity_num)))
    
       
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
     

    def __get_pages (self,
                     partition_mmapped,
                     offset_list: Iterable[int]) -> List[Page]:
        """
        Extract all the Page's from a CBOR partition file.
        It returns the list of pages in a partition.
        (Generators are not thread-safe).
        
        :param partition_id the mmapped stream of the partition
        :param offset_list the list of page offsets within the partition
        """
        
        pages = []
        for idx in offset_list:
            loaded_cbor = cbor.loads(partition_mmapped[idx:])
            pages.append(Page.from_cbor(loaded_cbor))

        return pages
        #return [Page.from_cbor(next(cbor.load(partition_mmapped[idx:])) for idx in offset_list)]
        #yield from [Page.from_cbor(cbor.loads(partition_mmapped[idx:]) for idx in offset_list)


    def __extract_links_monothreaded(self,
                                     partition_mmapped,
                                     offset_list: Iterable[int]) -> torch.sparse.LongTensor:
        """
        Calculate the frequency of each mention in wikipedia.
        
        :param partition_id the partition to use
        :param offset_list the list of offset within that partition
        :returns a sparse torch tensor
        """
        
        pages = self.__get_pages(partition_mmapped, offset_list)
        
        for page in pages:
            # remove spurious None elements
            if page is None:
                continue
            
            links = self.tokenize(page, False)
            
        freqs = Counter(links)

        keys = np.array([[0, k] for k in freqs.keys()]).T.reshape(2, -1)
        values = np.fromiter(freqs.values(), dtype=np.int32)

        return torch.sparse_coo_tensor(keys, values,
                                             size=(1, len(self)))
    

    def extract_links(self,
                        num_partitions=None,
                        partition_page_lim=1000,
                        pages_per_worker=100) -> torch.LongTensor:
        """
        Create some page batches and count mention occurrences for each batch.
        Summate results.

        This method is threaded (hopefully for the common good).
        
        :param num_partitions use only the given partitions. Useful for debugging
        :param partition_page_lim the maximum number of pages to take in a partition (None for all)
        :param pages_per_worker the number of pages to assign to a worker.
        """
            
        if num_partitions is None:
            num_partitions = len(self.partition_offsets)

       
        offset_lists = []
        mmapped = []
        for partition_id in tqdm.trange(num_partitions, desc="Reading partition tocs"):
            with open(os.path.join(self.partition_path, f"{partition_id}.cbor.toc")) as f:
                offset_lists.append([int(line.strip()) for line in f.readlines()])

            with open(os.path.join(self.partition_path, f"{partition_id}.cbor")) as f:
                mmapped.append(mmap.mmap(f.fileno(), 0, flags=mmap.MAP_PRIVATE))
            
        
        starting_tensor = torch.sparse.LongTensor(1, len(self))
        tensors = []
        with futures.ThreadPoolExecutor() as executor:
            promises = []

            
            if partition_page_lim is None:
                _partition_page_lim = len(offset_lists[partition_id])
            else:
                _partition_page_lim = min([partition_page_lim, len(offset_lists[partition_id])])

            # We create a number of workers and round-robin on each partition.
            # list transpose does the trick
            args_list = []
            for partition_id in range(num_partitions):
                args_list.append([])
                for idx in range(0, _partition_page_lim, pages_per_worker):
                    chosen_page_offsets = offset_lists[partition_id][idx:min(idx+pages_per_worker, _partition_page_lim)]
                    args_list[-1].append((self.__extract_links_monothreaded,
                                                mmapped[partition_id],
                                                chosen_page_offsets))
                    #tensors.append(self.__extract_links_monothreaded(partition_id, chosen_page_offsets))

            # https://stackoverflow.com/a/6473724
            args_list = map(list, zip(*args_list))

            # Flatten the list, and actually send the work
            for args in tqdm.tqdm(list(itertools.chain(*args_list)), desc="Scheduling work batches..."):
                promises.append(executor.submit(*args))

            for promise in tqdm.tqdm(futures.as_completed(promises), desc="Merging tokenized text"):
                tensors.append(promise.result())
        
        # cleanup before merging
        for mmapped_ in mmapped:
            mmapped_.close()

        # Stack the tensors, get rid of the dummy singleton (0) and sum column-wise (1)
        # Squeezes are not possible as sparse tensors have no strides
        return torch.sparse.sum(torch.stack(tensors), [0, 1]).to_dense()
    
    
    def __len__(self):
        return len(self.offsets)

    
    def tokenize(self,
                 page: Page,
                 tokenize=True) -> Union[List[int], Tuple[List[int], List[int]]]:
        """
        Tokenize a given page. Perform tree traversal over a Page structure
        and return the text and link tokens that constitute a page.

        :param page the page to tokenize
        :param tokenize if True, tokenize both text and links,
        otherwise just extract the links (with no padding)

        :return if `tokenize` is True then return a pair of token lists.
            Otherwise, return single token list
        TODO: add BIO support
        TODO: allow to access more than the first `self.token_length` tokens.
        """

        toks = []
        links = []

        # Encode a link. Cast to padding if the link was not "common".
        # Call this method only after preprocessing has been done!
        def encode_link(link):
            return self.key_restrictor.get(self.key_encoder.get(link, 0), 0)
            #return self.key_encoder.get(link, 0)

        # People say pattern matching is overrated.
        # I beg to differ.
        # (It's also true that a tree structure for tokenization makes
        # absolutely no sense - but I don't get to decide things apparently).
        def handle_section(skel: Section):
            for subskel in skel.children:
                if len(toks) >= self.token_length:
                    return
                visit_section(subskel)

        def handle_list(skel: ParaList):
            visit_section(skel.body)

        def handle_para(skel: Para):
            paragraph = skel.paragraph
            bodies = paragraph.bodies

            for body in bodies:
                visit_section(body)

        def handle_paratext(body: ParaBody):
            if tokenize:
                lemmas = self.tokenizer.tokenize(body.get_text())
                lemmas = self.tokenizer.convert_tokens_to_ids(lemmas)

                if len(lemmas) + len(toks) > self.token_length:
                    lemmas = lemmas[:(self.token_length - len(toks))]

                toks.extend(lemmas)
                links.extend([self.key_encoder.get("PAD")] * len(lemmas))

        def handle_paralink(body: ParaLink):
            if tokenize:
                lemmas = self.tokenizer.tokenize(body.get_text())
                lemmas = self.tokenizer.convert_tokens_to_ids(lemmas)

                if len(lemmas) + len(links) > self.token_length:
                    lemmas = lemmas[:(self.token_length - len(links))]

                toks.extend(lemmas)
                link_id = encode_link(body.page)

                if link_id in self.valid_keys:
                    links.extend([link_id] + [0] * (len(lemmas) - 1))
                else:
                    links.extend([0] * len(lemmas))
            else:
                links.append(self.key_encoder.get(body.page, 0))
            pass

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
        
        if tokenize:
            toks = pad_sequences([toks], maxlen=self.token_length, dtype="long",
                                 value=0.0, truncating="post", padding="post")

            links = pad_sequences([links], maxlen=self.token_length, dtype="long",
                                 value=0.0, truncating="post", padding="post")
            return toks[0], links[0]
        else:
            return links
       
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
    
    def __getitem__(self, idx):
        """
        Return a tensor with the given batches. The shape of a batch is
        (b x 3 x MAX_LEN).
        
        The first row is the token embedding via WordPiece ids.
        The second row is the BERT attention mask.
        The third row is the expected Entity Linking output.
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        elif type(idx) != list:
            idx = [idx]


        # Perform memory mapping
        pages = []
        for i in idx:
            k = self.offsets[i]
            partition_id, offset = self.key2partition(k)
            with wrap_open(os.path.join(self.partition_path, f"{partition_id}.cbor"), "rb") as partition_file:
                with mmap.mmap(partition_file.fileno(), 0, flags=mmap.MAP_PRIVATE) as partition_mmapped:
                    pages.extend(self.__get_pages(partition_mmapped, [offset]))
        
        
        toks_list = []
        attns_list = []
        links_list = []

        # Can we do this more efficiently?
        for page in pages:
            toks, links = self.tokenize(page)
            attns = self.get_attention_mask(toks)
            toks_list.append(toks)
            attns_list.append(attns)
            links_list.append(links)

        # TODO: possibly change the float format to float16...
        # Does the change happen when moving to GPU?
        # Need to investigate this...
        toks_list = torch.LongTensor(toks_list).squeeze()
        attns_list = torch.FloatTensor(attns_list).squeeze()
        links_list = torch.LongTensor(links_list).squeeze()

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