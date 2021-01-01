#!/bin/env python

from tools.dataloaders import WikipediaCBOR

wikipedia_cbor = WikipediaCBOR("wikipedia/car-wiki2020-01-01/enwiki2020.cbor",
                                "wikipedia/car-wiki2020-01-01/partitions")
wikipedia_cbor.preprocess(limit=100)
freqs = wikipedia_cbor.count_frequency()
print(freqs)
