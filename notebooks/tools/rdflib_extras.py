"""
Some extra function that aid parsing RDF ontologies
"""
from ontospy import Ontospy
from .sparql_wrapper import SPARQLDataProviders
import re

# TODO: add test cases
def extract_domain_codomain(ontology_filename: str):
    parsed_ontology = Ontospy(ontology_filename)
    properties = parsed_ontology.all_properties

    # To make this simple, every property has only one domain and one range
    # We ignore OWL intersections
    # We also consider the unconstrained relationships only
    parse_somevaluefrom = re.compile(r"owl:someValuesFrom <(.*?)>")
    
    property_dict = {}
    
    for owl_property in properties:
        source = [s.strip() for s in owl_property.rdf_source().split("\n")]

        parsing_domain = False
        parsing_range = False

        domain_property = ""
        range_property = ""

        for line in source:
            parsing_domain = parsing_domain or "rdfs:domain" in line
            parsing_range = parsing_range or "rdfs:range" in line

            parse_value = parse_somevaluefrom.match(line)
            if parse_value:
                to_add = parse_value.group(1)
                if parsing_domain:
                    domain_property = to_add
                elif parsing_range:
                    range_property = to_add
            
            if "]" in line:
                if parsing_domain:
                    parsing_domain = False

                elif parsing_range:
                    parsing_range = False

        if domain_property == "":
            continue
        
        # recursive properties seem to only specify domain properties
        if range_property == "":
            range_property = domain_property
        
        stripped_property, stripped_domain, stripped_range = [SPARQLDataProviders.strip_namespace(s) for s in
            [str(owl_property.uri), domain_property, range_property]]

        property_dict[stripped_property] = (stripped_domain, stripped_range)

    return property_dict


if __name__ == "__main__":
    print(extract_domain_codomain("../ontology.owl"))