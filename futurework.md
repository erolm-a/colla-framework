# Future Work


## On dataset collection

We are using Pyserini for harvesting questions from GoogleNLQ as our sole data source.
It would be nice to inspect other data sources (if available), maybe launching a
commercial version of this product and collect user feedback.

## On KG generation

The current KG relies on the correctness of Wiktionary and BabelNet. The former one is a voluntary project, thus it's expectable many results may be broken and/or incomplete and/or inconsistent. For example, we glossed the problem of repeated word entries by just squashing the senses.

The quality of the extraction is heavily dependent on Tatu Ylonen's wiktextract project. We wrote a PR to improve the existing support so that senses, subsenses and usages were captured, but so far they are underused in the final KG. The same applies to a number of fields not used in the final RDF dump like IPA pronunciation or target translations. The latter is indeed paramount for a linguistic KG and will be likely among the most prioritized tasks.

Other technical problems include:

- lack of form support for non-English lexemes. It seems Tatu Ylonen released support for form generation via Lua template script sandboxing a few weeks after we wrote a custom generator (which is a mere Python port of the lua script)

## On utterance parsing

The current grammar is obviously a loose baseline for a potentially more powerful query rewriter.

One could try to:

- Add ML on the grammar rather than relying on a potentially forgetful and/or bugged greedy picker.
- Finetune a seq2seq model that translated into classical intent-slots, for example finetune a pretrained BART on linguistic question detection and translation.
- Scrap the intent system and use a more powerful query rewriter system. Approaches like this are hardly introspectable and require massive datasets, so we assume this won't happen in the near future.

## On intent-query translation

The main issue with the current system is that queries are performed via SPARQL. Thus, slots need to be carefully translated every time. On top of that, they are quite fragile (e.g. searching for a specific form requires the form slot to perfectly match with the form label in the KG - but internal tokenization may break this).

Also, since it is objectively impossible to create a variant of a SPARQL rule for each slot combination, we adopted a 'catch-it-all' strategy where the query is generic enough and results are programmatically picked so that they match the given slots.

There are two possibilities ahead:

- Preserve SPARQL, but adopt an automated slot-to-SPARQL translation mechanism. This will move the complexity of user queries to SPARQL.
- Deprecate SPARQL and use another constraint based KG explorer. It was proposed to abandon LOD-style knowledge graphs in favour of other graphs with simpler schemas. The neo4j project seems interesting on that regard and should be explored, despite it shares similar issues.