import pke
import json
import string
import argparse
from tqdm import tqdm
import re
import spacy
from spacy.tokenizer import _get_regex_pattern

nlp = spacy.load("en_core_web_sm")

# Tokenization fix for in-word hyphens (e.g. 'non-linear' would be kept 
# as one token instead of default spacy behavior of 'non', '-', 'linear')
# https://spacy.io/usage/linguistic-features#native-tokenizer-additions

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

parser = argparse.ArgumentParser()

parser.add_argument("-i",
                    "--input")

parser.add_argument("-o",
                    "--output")

args = parser.parse_args()

print("Input file: {}".format(args.input))
print("Output file: {}".format(args.output))

# load reference file
documents = []
with open(args.input, 'r') as f:
    for i, line in enumerate(f):
        documents.append(json.loads(line))

# populates with spacy doc objects
for i, doc in enumerate(tqdm(documents)):
    text = doc["title"].strip()
    if not text[-1] in string.punctuation:
        text += "."
    text += " " + doc["abstract"].strip()
    documents[i]["spacy"] = nlp(text)

extractor = pke.unsupervised.MultipartiteRank()
pred = []
for doc in tqdm(documents):
    
    extractor.load_document(input=doc["spacy"], language='en')
    
    #extractor.candidate_selection()
    extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")
    extractor.candidate_filtering()
    
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=10)
    pred.append({"id": doc["id"], "top_m": "", "top_k": ";".join([k for k, v in keyphrases])})

with open(args.output, 'w') as f:
    f.write("\n".join([json.dumps(doc) for doc in pred]))