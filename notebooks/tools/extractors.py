"""
A collection of Wiktionary template extractors
"""

import re
from pyspark.sql.functions import udf, struct, explode
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, BooleanType
from functools import reduce


def english_verb_form_extractor(lexeme, head):
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


def make_comparative(lexeme, comp_sup_list):
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
    

def english_adjective_form_extractor(lexeme, head):
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
        optional = {'optional': '"generally not comparable"'}
    
    return {**make_comparative(lexeme, params), **optional}

def english_adverb_form_extractor(lexeme, head):
    # The two take virtually the same parameters and behave in the same way
    return english_adjective_form_extractor(lexeme, head)

def english_noun_form_extractor(lexeme, head):
    plurals = []
    # sometimes the head parameters are given to "plxqual" rather than just as
    # number parameter of the head
    for i in range(1, 15):
        pl = head[str(i)]
        if pl:
            qual = head["pl" + ("" if i == 1 else str(i)) + "qual"]
            if qual:
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
    
    print(plurals)
    
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

def extract_form(cursor):
    # assuming the cursor only works on lexemes (FIXME)
    # and entries are English-only (FIXME)
    udf_en_verbs = udf(lambda row: english_verb_form_extractor(*row), english_verb_schema)
    udf_en_adjs = udf(lambda row: english_adjective_form_extractor(*row), english_adjective_schema)
    udf_en_advs = udf(lambda row: english_adverb_form_extractor(*row), english_adverb_schema)
    udf_en_noun = udf(lambda row: english_noun_form_extractor(*row), english_noun_schema)
    
    template_col = cursor["head"].template_name.alias("template")
    word_col = cursor.word
    extracted_struct = struct([cursor.word, 'head'])
    
    cursor = cursor.select([*cursor.columns, template_col ])
    
    def get_pos_df(pos, udf):
        return cursor.where(f'head.template_name = "en-{pos}"') \
                        .select([word_col.alias('word'), template_col,
                                   udf(extracted_struct).alias(f'{pos}_forms')])
    
    verbs_df = get_pos_df("verb", udf_en_verbs)
    adjs_df = get_pos_df("adj", udf_en_adjs)
    advs_df = get_pos_df("adv", udf_en_advs)
    noun_df = get_pos_df("noun", udf_en_noun)
    
    def custom_join(df1, df2):
        return df1.join(df2, ['word', 'template',], "leftouter")
    
    return reduce(custom_join, [verbs_df, adjs_df, advs_df, noun_df], cursor)

def extract_df(dataframe, word = None):
    """Explode the heads of the entry, and potentially filter by word"""

    current_cursor = dataframe.withColumn("head", explode("heads")).drop("heads")

    if word:
        current_cursor = current_cursor.where(dataframe.word == word)
        
    return current_cursor