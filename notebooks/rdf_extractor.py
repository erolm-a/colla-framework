#!/usr/bin/env python3

import argparse
import subprocess

import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession

from tools.dumps import get_filename_path
from tools.providers import WiktionaryProvider, WiktionaryTV, WiktionaryProjectGutenberg
from tools.extractors import extract_df, extract_form, convert_to_jsonl
from tools.rdf_dumping import ExtractorGraph


def main():
    parser = argparse.ArgumentParser(description="ColLa RDF extractor")
    parser.add_argument("-w", "--wiktionary-revision", help="the wikitionary pages-article.xml revision to use.",
        type=str, required=True)
    parser.add_argument("-l", "--wordlist", help="A wordlist to use. If not provided, perform extraction on the whole dictionary.",
        choices=["wdtv", "wdpg"])
    parser.add_argument("-o", "--output", help="The output file to generate a dump to. The format is Turtle",
        type=str, required=True)
    parser.add_argument("-s", "--skip-wiktionary", help="Skip wiktionary dumping (for debug purposes, may cause unreproducible results!)",
        action="store_true")

    
    args = parser.parse_args()
    revision = args.wiktionary_revision
    wordlist_arg = args.wordlist

    if wordlist_arg == "wdpg":
        wordlist = WiktionaryProjectGutenberg.get_wordlist()
    elif wordlist_arg == "wdtv":
        wordlist = WiktionaryTV().get_wordlist()
    else:
        wordlist = None
    
    output = get_filename_path(args.output)

    wiktextract_output_file = get_filename_path(f"wiktionary/{revision}-English.json")

    if not args.skip_wiktionary:
        print(f"Beginning to dump the Wiktionary revision {revision}")
        WiktionaryProvider.dump_full_dataset(revision=revision)
        
        print(f"Beginning Wiktionary parsing. This will take a while")
        wiktionary_output_file = get_filename_path(f"wiktionary/enwiktionary-{revision}-pages-articles.xml.bz2")
        subprocess.run(["wiktwords", wiktionary_output_file, "--out", wiktextract_output_file, "--language", "English", "--all"])

    print("Beginning wiktionary extraction and entity linking")

    sc = SparkContext()
    spark = SparkSession(sc)

    wiktionary_df = spark.read.json(wiktextract_output_file)
    # Filter by the given wordlist, if given
    if wordlist:
        wordlist_df = spark.createDataFrame(pd.DataFrame({'word': wordlist}))
        cursor = wiktionary_df.join(wordlist_df, "word", "inner")
    
    else:
        cursor = wiktionary_df
    
    forms = extract_form(extract_df(cursor))

    output_graph = ExtractorGraph()

    print("Extracting forms into the graph...")
    for row in forms.rdd.toLocalIterator():
        if(row['head']['template_name'] != 'head'):
            output_graph.add_wiktionary(row)

    print(f"Serializing graph into {output}")
    output_graph.rdflib_graph.serialize(output)
    

if __name__ == "__main__":
    main()
