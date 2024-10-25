import json
from tqdm import tqdm
import argparse
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np

stemmer = PorterStemmer()
PAD_PHRASE = "<<bad-bad>>"
PAD_MIN = 10

parser = argparse.ArgumentParser()

parser.add_argument("-r",
                    "--reference")

parser.add_argument("-s",
                    "--system")

parser.add_argument('--output_scores', 
                    action='store_true')

args = parser.parse_args()

def tokenize(s):
    """Tokenize an input text."""
    return word_tokenize(s)

def lowercase_and_stem(_words):
    """lowercase and stem sequence of words."""
    return [stemmer.stem(w.lower()) for w in _words]

def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))

def preprocess_phrases(phrases):
    pre_phrases = [' '.join(lowercase_and_stem(tokenize(phrase))) for phrase in phrases]
    #print("B", pre_phrases)
    # remove duplicate
    pre_phrases = list(dict.fromkeys(pre_phrases))
    # remove empty phrases
    pre_phrases = list(filter(None, pre_phrases))
    #print("A", pre_phrases)
    return pre_phrases
    #return [' '.join(lowercase_and_stem(phrase.split(" "))) for phrase in phrases]

def evaluate(top_N_keyphrases, references):
    if not len(top_N_keyphrases):
        return (0.0, 0.0, 0.0)
    P = len(set(top_N_keyphrases) & set(references)) / len(top_N_keyphrases)
    R = len(set(top_N_keyphrases) & set(references)) / len(references)
    F = (2*P*R)/(P+R) if (P+R) > 0 else 0
    return (P, R, F)

# load output file
top_m = {}
top_k = {}
with open(args.system, 'r') as f:
    for line in f:
        doc = json.loads(line)
        top_m[doc["id"]] = preprocess_phrases(doc["top_m"].split(";"))
        top_k[doc["id"]] = preprocess_phrases(doc["top_k"].split(";"))
        top_k[doc["id"]].extend([PAD_PHRASE for i in range(PAD_MIN-len(top_k[doc["id"]]))])

# load reference file
references = {}
tgt_pres_abs = defaultdict(list)
pre_pres_abs_top_m = defaultdict(list)
pre_pres_abs_top_k = defaultdict(list)
with open(args.reference, 'r') as f:
    for line in f:
        doc = json.loads(line)
        
        # keywords / keyphrases switch
        if "keywords" in doc:
            keyphrases = doc["keywords"].split(";")
        else:
            keyphrases = doc["keyphrases"]

        # preprocess keyphrases
        references[doc["id"]] = preprocess_phrases(keyphrases)

        # preprocess title and abstract
        title = lowercase_and_stem(tokenize(doc["title"]))
        abstract = lowercase_and_stem(tokenize(doc["abstract"]))

        # check for present / absent keyphrases in references (tgt)
        for keyphrase in references[doc["id"]]:
            tokens = keyphrase.split(" ")
            tgt_pres_abs[doc["id"]].append(0)
            if contains(tokens, title) or contains(tokens, abstract):
                tgt_pres_abs[doc["id"]][-1] = 1

        # check for present / absent keyphrases in top_m predicted (pre)
        for keyphrase in top_m[doc["id"]]:
            tokens = keyphrase.split(" ")
            pre_pres_abs_top_m[doc["id"]].append(0)
            if contains(tokens, title) or contains(tokens, abstract):
                pre_pres_abs_top_m[doc["id"]][-1] = 1

        # check for present / absent keyphrases in top_k predicted (pre)
        for keyphrase in top_k[doc["id"]]:
            tokens = keyphrase.split(" ")
            pre_pres_abs_top_k[doc["id"]].append(0)
            if contains(tokens, title) or contains(tokens, abstract):
                pre_pres_abs_top_k[doc["id"]][-1] = 1


# loop through the documents
scores_at_m = defaultdict(list)
scores_at_5 = defaultdict(list)
scores_at_10 = defaultdict(list)
valid_keys =  defaultdict(list)
for i, docid in enumerate(references):

    # compute scores for all references
    scores_at_m['all'].append(evaluate(top_m[docid], references[docid]))
    scores_at_5['all'].append(evaluate(top_k[docid][:5], references[docid]))
    scores_at_10['all'].append(evaluate(top_k[docid][:10], references[docid]))
    valid_keys['all'].append(docid)

    # add scores for present and absent keyphrases
    pres_references = [phrase for j, phrase in enumerate(references[docid]) if tgt_pres_abs[docid][j]]
    pres_top_m = [phrase for j, phrase in enumerate(top_m[docid]) if pre_pres_abs_top_m[docid][j]]
    pres_top_k = [phrase for j, phrase in enumerate(top_k[docid]) if pre_pres_abs_top_k[docid][j]]
    pres_top_k.extend([PAD_PHRASE for j in range(PAD_MIN-len(pres_top_k))])
    if len(pres_references):
        scores_at_m['pre'].append(evaluate(pres_top_m, pres_references))
        scores_at_5['pre'].append(evaluate(pres_top_k[:5], pres_references))
        scores_at_10['pre'].append(evaluate(pres_top_k[:10], pres_references))
        valid_keys['pre'].append(docid)

    abs_references = [phrase for j, phrase in enumerate(references[docid]) if not tgt_pres_abs[docid][j]]
    abs_top_m = [phrase for j, phrase in enumerate(top_m[docid]) if not pre_pres_abs_top_m[docid][j]]
    abs_top_k = [phrase for j, phrase in enumerate(top_k[docid]) if not pre_pres_abs_top_k[docid][j]]
    abs_top_k.extend([PAD_PHRASE for j in range(PAD_MIN-len(pres_top_k))])
    if len(abs_references):
        scores_at_m['abs'].append(evaluate(abs_top_m, abs_references))
        scores_at_5['abs'].append(evaluate(abs_top_k[:5], abs_references))
        scores_at_10['abs'].append(evaluate(abs_top_k[:10], abs_references))
        valid_keys['abs'].append(docid)

# compute the average scores
for eval in ['all', 'pre', 'abs']:
    avg_scores_at_m = np.mean(scores_at_m[eval], axis=0)
    avg_scores_at_5 = np.mean(scores_at_5[eval], axis=0)
    avg_scores_at_10 = np.mean(scores_at_10[eval], axis=0)
            
    # print out the performance of the model
    print("{} F@M: {:>4.1f} F@5: {:>4.1f} F@10: {:>4.1f} - {}".format(eval, avg_scores_at_m[2]*100, avg_scores_at_5[2]*100, avg_scores_at_10[2]*100, args.system.split("/")[-1]))

    if args.output_scores:
        output_file = re.sub("\.jsonl$", "", args.system) + ".{}.csv".format(eval)
        with open(output_file, 'w') as f:
            for i, docid in enumerate(valid_keys[eval]):
                f.write("{}\t{}\t{}\t{}\n".format(docid, scores_at_m[eval][i][2], scores_at_5[eval][i][2], scores_at_10[eval][i][2]))



























