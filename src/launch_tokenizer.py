#!/bin/env python
"""
FIXME: delete me as soon as debugging is over
"""

from tools.dataloaders import WikipediaCBOR, SQuADDataloader
from concurrent.futures import ThreadPoolExecutor, as_completed

from itertools import zip_longest # for Python 3.x

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import os


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def get_slices(wikipedia_cbor, slices):
    return [wikipedia_cbor[i] for i in slices]


def main():
    
    wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor",
                                    "wikipedia/car-wiki2020-01-01/partitions",
                                    page_lim=1000, clean_cache=True
                                    )

    """

    bs = 64
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    # use only 0.1%  of our dataset for debugging purposes.
    # Yes, I should make better efforts for making a development dataset
    wiki_use_size = int(0.1 * len(wikipedia_cbor))
    wikipedia_cbor_limited, _ = random_split(wikipedia_cbor,
                                            [wiki_use_size, len(wikipedia_cbor) - wiki_use_size],
                                            generator=torch.Generator().manual_seed(42))

    wiki_train_size = int(0.8*len(wikipedia_cbor_limited))
    wiki_validation_size = len(wikipedia_cbor_limited) - wiki_train_size

    wikipedia_cbor_train, wikipedia_cbor_validation = random_split(wikipedia_cbor_limited,
                                        [wiki_train_size, wiki_validation_size],
                                        generator=torch.Generator().manual_seed(42))

    wiki_train_sampler = RandomSampler(wikipedia_cbor_train, generator=torch.Generator().manual_seed(42))
    wiki_train_dataloader = DataLoader(wikipedia_cbor_train, sampler=wiki_train_sampler, batch_size=bs, num_workers=2)

    #wiki_validation_sampler = RandomSampler(wikipedia_cbor_validation)
    # wiki_validation_dataloader = DataLoader(wikipedia_cbor_validation, sampler=wiki_validation_sampler, batch_size=bs, num_workers=8)
       
    
    for idx, batch in enumerate(wiki_train_dataloader):
        print("Loaded batch")
        print(len(batch))

        if idx >= 10:
            break
        


    # wikipedia_cbor.tokenizer.
    """

    squad_dataset = SQuADDataloader()
    train_dataset = squad_dataset.train_dataset

    for b in train_dataset:
        print(b)
        break
    
if __name__ == "__main__":
    main()
