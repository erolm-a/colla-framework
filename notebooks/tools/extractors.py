"""
A collection of Wiktionary template extractors.

Nota Bene: as of Aug 19, 2020 Tatu Ylonen has finished writing a form extractor
within wiktionary. Thus, all the form extractors below are to be deprecated.
"""

import re
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.functions import col, explode, struct, udf
from pyspark.sql.types import (ArrayType, StringType, StructType, StructField,
                               BooleanType, DataType)
from functools import reduce
from .strings import hash
from typing import Any, Dict, List, Tuple, Optional, Union

def english_verb_form_extractor(lexeme: str, head: Column) -> Dict[str, Any]:
    # Decoding rules (abridged from https://en.wiktionary.org/wiki/Template:en-verb):
    # Ported from Lua (https://en.wiktionary.org/wiki/Module:en-headword)
    
    par1 = head['1']
    par2 = head['2']
    par3 = head['3']
    par4 = head['4']
    
    pres_3sg_form = par1 or lexeme + "s"
    pres_ptc_form = par2 or lexeme + "ing"
    past_form = par3 or lexeme + "ed"
    
    if par1 and not par2 and not par3:
        # "new" format, which only uses the first parameter
        if par1 == "es":
            pres_3sg_form = lexeme + "es"
            pres_ptc_form = lexeme + "ing"
            past_form = lexeme + "ed"
        # strip -y, add -ies, -ied, -ying
        elif par1 == "ies":    
            stem = lexeme[:-1]
            pres_3sg_form = stem + "ies"
            pres_ptc_form = stem + "ying"
            past_form = stem + "ied"
        # verb takes a single -d in the past tense
        elif par1 == "d":
            pres_3sg_form = lexeme + "s"
            pres_ptc_form = lexeme + "ing"
            past_form = lexeme + "d"
        # e.g. {{en-verb|admir}}
        else:
            pres_3sg_form = lexeme + "s"
            pres_ptc_form = par1 + "ing"
            past_form = par1 + "ed"
    else:
        # "legacy" format, that uses the second and the third parameter as well
        if par3:
            if par3 == "es":
                pres_3sg_form = par1 + par2 + "es"
                pres_ptc_form = par1 + par2 + "ing"
                past_form = par1 + par2 + "ed"
            elif par3 == "ing":
                pres_3sg_form = lexeme + "s"
                pres_ptc_form = par1 + par2 + "ing"
                if par2 == "y":
                    past_form = lexeme + "d"
                else:
                    past_form = par1 + par2 + "ed"
            elif par3 == "ed":
                if par2 == "i":
                    pres_3sg_form = par1 + par2 + "es"
                    pres_ptc_form = lexeme + "ing"
                else:
                    pres_3sg_form = lexeme + "s"
                    pres_ptc_form = par1 + par2 + "ing"
            elif par3 == "d":
                pres_3sg_form = lexeme + "s"
                pres_ptc_form = par1 + par2 + "ing"
                past_form = par1 + par2 + "d"
        else:
            if par2 == "es":
                pres_3sg_form = par1 + "es"
                pres_ptc_form = par1 + "ing"
                past_form = par1 + "ed"
            elif par2 == "ies":
                pres_3sg_form = par1 + "ies"
                pres_ptc_form = par1 + "ying"
                past_form = par1 + "ied"
            elif par2 == "ing":
                pres_3sg_form = lexeme + "s"
                pres_ptc_form = par1 + "ing"
                past_form = par1 + "ed"
            elif par2 == "ed":
                pres_3sg_form = lexeme + "s"
                pres_ptc_form = par1 + "ing"
                past_form = par1 + "ed"
            elif par2 == "d":
                pres_3sg_form = lexeme + "s"
                pres_ptc_form = par1 + "ing"
                past_form = par1 + "d"

    past_ptc_forms = par4 or past_form
    return {"lexeme": lexeme, "pres_3sg": pres_3sg_form,
            "pres_ptc": pres_ptc_form, "past": past_form,
            "past_ptc": past_ptc_forms}


def make_comparative(lexeme: str, comp_sup_list: List[Tuple[str, str]]) \
    -> Dict[str, Union[str, List[str]]]:
    if len(comp_sup_list) == 0:
        comp_sup_list.append(("more", None))
    
    # extract the stem
    # If ending with -(e)y, replace with -i
    # If ending with -e, remove the -e
    stem = re.sub("e$", "", re.sub("([^aeiou])e?y$", r"\1i", lexeme))
    comp_forms = []
    sup_forms = []
        
    for comp, sup in comp_sup_list:
        if comp == "more" and lexeme != "many" and lexeme != "much":
            comp_forms.append("more " + lexeme)
            sup_forms.append("most " + lexeme)
        elif comp == "further" and lexeme != "far":
            comp_forms.append("further " + lexeme)
            sup_forms.append("furthest " + lexeme)
        elif comp == "er":
            comp_forms.append(stem + "er")
            sup_forms.append(stem + "est")
        elif comp == "-" or sup == "-":
            if comp != "-":
                comp_forms.append(comp)
            if sup != "-":
                sup_forms.append(sup)
        else:
            if not sup:
                if comp.endswith("er"):
                    sup = comp[:-2] + "est"
            comp_forms.append(comp)
            sup_forms.append(sup)
    
    return {"lexeme": lexeme, "comparatives": comp_forms, "superlatives": sup_forms}
    

def english_adjective_form_extractor(lexeme: str, head: Column) -> Dict[str, Any]:
    shift = 0
    is_not_comparable = False
    is_comparable_only = False
    
    if head["1"] == '?':
        return

    if head["1"] == '-':
        shift = 1
        is_not_comparable = True
    elif head["1"] == "+":
        shift = 1
        is_comparable_only = True
    
    # Empirically, adjectives can have up to 4 superlatives.
    params = []
    
    for i in range(1, 5):
        comp = head[str(i + shift)]
        sup = head[f"sup{i}"]
        if comp or sup:
            params.append((comp, sup))
    
    optional = {}
    
    if shift == 1:
        if len(params) == 0:
            if is_not_comparable:
                return {"lexeme": lexeme, "optional": "not comparable"}
            if is_comparable_only:
                return {"lexeme": lexeme, "optional": "comparable-only"}
        optional = {'optional': "generally not comparable"}
    
    return {**make_comparative(lexeme, params), **optional}

def english_adverb_form_extractor(lexeme: str, head: Column):
    # Adjective and adverb templates virtually the same parameters and behave in the same way
    return english_adjective_form_extractor(lexeme, head)

def english_noun_form_extractor(lexeme: str, head: Column) -> Dict[str, Any]:
    plurals = []
    # sometimes the head parameters are given to "plxqual" rather than just as
    # number parameter of the head
    for i in range(1, 15):
        pl = head[str(i)]
        if pl:
            qual_name = "pl" + ("" if i == 1 else str(i)) + "qual"
            if qual_name in head and head[qual_name]:
                qual = head[qual_name]
                plurals.append({'term': 'pl', 'qualifiers': [qual]})
            else:
                plurals.append(pl)
    
    # Handle special plurals (defective, uncountable, unknown ...)
    mode = None
    if len(plurals) > 0 and plurals[0] in ["?", "!", "-", "~"]:
        mode = plurals[0]
        plurals = plurals[1:]
    

    optional = {"countable": "yes", "irregular": False}
    if mode == "?":
        optional.update({"countable": "no", "optional": "uncertain"})
        return optional
    
    elif mode == "!":
        optional.update({"countable": "no", "optional": "unattested"})
        return optional
    
    elif mode == "-": # uncountable noun, may have a plural
        optional.update({"countable": "no", "always": not len(plurals)})
    
    elif mode == "~": # mixed countable/uncountable. Always has a plural
        optional.update({"countable": "sometimes"})
        if len(plurals) == 0:
            plurals = ["s"]
    else:
        if len(plurals) == 0:
            plurals = ["s"]
    
    if len(plurals) == 0:
        return optional
    
    # replace -y with -ies if -y is preceded by consonant
    def check_ies(pl, stem):
        new_plural = re.sub("([^aeiou])y$", r"\1ies", stem)
        return new_plural != stem and pl == new_plural
    
    stem = lexeme
    final_plurals = []
    
    for pl in plurals:
        if pl == "s":
            final_plurals.append(stem + "s")
        elif pl == "es":
            final_plurals.append(stem + "es")
        else:
            if isinstance(pl, dict):
                pl = pl['term']
            final_plurals.append(pl)
            if not " " in stem and not (pl == stem + "s" or pl == stem + "es"
                                            or check_ies(pl, stem)):
                optional["irregular"] = True
                # The original code also checks if a noun is "undeclinable".
                # We can safely ignore that.
    
        
    return {'plurals': final_plurals, **optional}

english_verb_schema = StructType([
    StructField("lexeme", StringType()),
    StructField("pres_3sg", StringType(), nullable=True), # defective verbs
    StructField("pres_ptc", StringType()),
    StructField("past", StringType()),
    StructField("past_ptc", StringType())
])

english_adjective_schema = StructType([
    StructField("lexeme", StringType()),
    StructField("comparatives", ArrayType(StringType()), nullable=True),
    StructField("superlatives", ArrayType(StringType()), nullable=True),
    StructField("optional", StringType(), nullable=True)
])

english_adverb_schema = english_adjective_schema

english_noun_schema = StructType([
    StructField("plurals", ArrayType(StringType())),
    StructField("countable", StringType()), # yes/no/sometimes
    StructField("irregular", BooleanType()),
    StructField("always", BooleanType(), nullable=True),
    StructField("optional", StringType(), nullable=True)
])

def udf_wrapper(func, schema: DataType, template):
    def func_wrapper(lexeme, head):
        if head['template_name'] == template:
            return func(lexeme, head)
    return udf(lambda row: func_wrapper(*row), schema)
    

def extract_form(cursor: DataFrame) -> DataFrame:
    # assuming the cursor only works on lexemes (FIXME)
    udf_verbs = udf_wrapper(english_verb_form_extractor, english_verb_schema, "en-verb")
    udf_adjs = udf_wrapper(english_adjective_form_extractor, english_adjective_schema, "en-adj")
    udf_advs = udf_wrapper(english_adverb_form_extractor, english_adverb_schema, "en-adv")
    udf_nouns = udf_wrapper(english_verb_form_extractor, english_noun_schema, "en-noun")
       
    for name, udf in [('verb_forms', udf_verbs), ('adj_forms', udf_adjs),
                         ('adv_forms', udf_advs), ('noun_forms', udf_nouns)]:
        
        extracted_struct = struct('word', 'head')
        cursor = cursor.withColumn(name, udf(extracted_struct))
    
    return cursor

def extract_df(dataframe: DataFrame, word: Optional[str] = None) -> DataFrame:
    """Explode the heads of the entry, and potentially filter by word"""

    current_cursor = dataframe.withColumn("head", explode("heads")).drop("heads")

    if word:
        current_cursor = current_cursor.where(dataframe.word == word)
        
    return current_cursor

def convert_to_jsonl(cursor: DataFrame, index_output: str):
    """
    Serialize a Wiktionary dataframe whose forms into a JSONL document that can
    be then processed by Anserini's JsonCollection
    """

    def text_representation(lexeme, pos, senses, verb_forms, adj_forms, noun_forms, adv_forms):
        """Generate a trivial text representation of the entry.
        This function is used as a UDF.
        
        For example:
        
        ```
        love (noun), id: kgl:1udqijwqaje2
        1. strong affection towards someone

        forms: love, loves
        ```
        """
        message = f"{lexeme} ({pos}), id: kgl:{hash(lexeme, pos)}\n"
    
        if senses:
            for idx, sense in enumerate(senses):
                message += f"{idx}. {sense['glosses'][0] if sense['glosses'] else ''}\n"

            if verb_forms:
                message += f"forms: {verb_forms['pres_ptc']}, {verb_forms['pres_3sg']}, {verb_forms['past']}, {verb_forms['past_ptc']}"
            if noun_forms and noun_forms['plurals']:
                message += f"forms: {', '.join(noun_forms['plurals'])}"
            if adj_forms:
                message += "forms:"
                if adj_forms.superlatives and all(adj_forms.comparatives):
                    message += "\ncomparatives:\n"
                    message += ", ".join(adj_forms.comparatives)
                if adj_forms.superlatives and all(adj_forms.superlatives):
                    message += "\nsuperlatives:\n"
                    message += ", ".join(adj_forms.superlatives)

            if adv_forms:
                message += "forms:"
                if adv_forms.comparatives and all(adv_forms.comparatives):
                    message += "\ncomparatives:\n"
                    message += ", ".join(adv_forms.comparatives)
                if adv_forms.superlatives and all(adv_forms.superlatives):
                    message += "\nsuperlatives:\n"
                    message += ", ".join(adv_forms.superlatives)

            return message


    udf_text_representation = udf(lambda row: text_representation(*row),
                                        StringType())
    udf_hash = udf(lambda row: hash(*row))

    trectext_wdpg = cursor.withColumn('word_id', udf_hash(struct('word',
                                                                 'pos')))
    trectext_wdpg = trectext_wdpg.withColumn('trectext_content',
                                udf_text_representation(
                                    struct('word', 'pos', 'senses',
                                           'verb_forms', 'adj_forms',
                                           'noun_forms', 'adv_forms')))
    trectext_wdpg.select([col('word_id').alias('id'),
                          col('trectext_content').alias('contents')]) \
                .toPandas().to_json(index_output, orient='records', force_ascii=False)