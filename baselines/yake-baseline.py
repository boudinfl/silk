import yake
import json
import string
import argparse
from tqdm import tqdm

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

kw_extractor = yake.KeywordExtractor(top=10)

pred = []
for doc in tqdm(documents):
    text = doc["title"].strip()
    if not text[-1] in string.punctuation:
        text += "."
    text += " " + doc["abstract"].strip()

    keyphrases = kw_extractor.extract_keywords(text)
    pred.append({"id": doc["id"], "top_m": "", "top_k": ";".join([k for k, v in keyphrases])})

with open(args.output, 'w') as f:
    f.write("\n".join([json.dumps(doc) for doc in pred]))
    