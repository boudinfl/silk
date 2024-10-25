import sys
import math
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset

parser = argparse.ArgumentParser()

parser.add_argument("-m",
                    "--model",
                    default="taln-ls2n/bart-kp20k")

parser.add_argument("--token",
                    default="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")

parser.add_argument("--num_beams",
                    default=20,
                    type=int)

parser.add_argument("--num_return_sequences",
                    default=20,
                    type=int)

parser.add_argument("--batch_size",
                    default=8,
                    type=int)

parser.add_argument("-i",
                    "--input")

parser.add_argument("-o",
                    "--output")

args = parser.parse_args()

print("Using model: {}".format(args.model))
print("Input file: {}".format(args.input))
print("Output file: {}".format(args.output))


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def join_titles_and_abstracts(dataset, special_token="<s>"):
    dataset["src"] = "{}<s>{}".format(dataset["title"], dataset["abstract"])
    return dataset

# Loading the dataset for testing
dataset = load_dataset("json", 
                       data_files=args.input,
                       split="train")

dataset = dataset.map(join_titles_and_abstracts)

#dataset = dataset.select(range(32))
#print(dataset)


tokenizer = AutoTokenizer.from_pretrained(args.model,
                                          token=args.token,
                                          model_max_length=512)

model = BartForConditionalGeneration.from_pretrained(args.model,
                                                     token=args.token)

pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer) #, device="mps")

#model = model.to("mps")

pred_phrases = []
for pred in tqdm(pipe(KeyDataset(dataset, "src"), 
                      batch_size=args.batch_size, 
                      truncation="only_first", 
                      max_length=40, 
                      num_beams=args.num_beams, 
                      num_return_sequences=args.num_return_sequences), total=len(dataset)):
    top_m = pred[0]['generated_text']
    top_k = []
    for sequence in pred:
        for phrase in sequence['generated_text'].split(";"):
            phrase = phrase.lower().strip()
            if phrase not in set(top_k):
                top_k.append(phrase)
    
    pred_phrases.append((dataset["id"][len(pred_phrases)], top_m, ";".join(top_k)))

with open(args.output, 'w') as f:
    f.write("\n".join([json.dumps({"id": res[0], "top_m": res[1], "top_k": res[2]}) for res in pred_phrases]))