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
                    default="models/bart-base")

parser.add_argument("--batch_size",
                    default=8,
                    type=int)

parser.add_argument("--cut_off",
                    default=2000,
                    type=int)

parser.add_argument("-i",
                    "--input")

parser.add_argument("-o",
                    "--output")

args = parser.parse_args()

print("Using model: {}".format(args.model))
print("Input file: {}".format(args.input))
print("Output file: {}".format(args.output))


input_dataset = []
with open(args.input, 'r') as f:
    for line in f:
        input_dataset.append(json.loads(line))

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
input_dataset = input_dataset[:args.cut_off]
dataset = dataset.select(range(args.cut_off))

print("Cut off for self-training samples: {}".format(len(dataset)))
#print(dataset)


tokenizer = AutoTokenizer.from_pretrained(args.model,
                                          model_max_length=512)

model = BartForConditionalGeneration.from_pretrained(args.model)

pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer) #, device="mps")

#model = model.to("mps")

pred_phrases = {}
for pred in tqdm(pipe(KeyDataset(dataset, "src"), 
                      batch_size=args.batch_size, 
                      truncation="only_first", 
                      max_length=40, 
                      num_beams=1, 
                      num_return_sequences=1), total=len(dataset)):
    pred_phrases[dataset["id"][len(pred_phrases)]] = pred[0]['generated_text'].split(";")

for i, sample in enumerate(input_dataset):
    input_dataset[i]["references"] = input_dataset[i]["keyphrases"]
    input_dataset[i]["keyphrases"] = pred_phrases[sample["id"]]


with open(args.output, "w") as f:
    f.write('\n'.join([json.dumps(doc) for doc in input_dataset]))

#with open(args.output, 'w') as f:
#    f.write("\n".join([json.dumps({"id": res[0], "top_m": res[1], "top_k": res[2]}) for res in pred_phrases]))