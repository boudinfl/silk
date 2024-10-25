import sys
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForSeq2Seq


parser = argparse.ArgumentParser()

parser.add_argument("-m",
                    "--model",
                    default="taln-ls2n/bart-kp20k")

parser.add_argument("--token",
                    default="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")

parser.add_argument("--batch_size",
                    default=16,
                    type=int)

parser.add_argument("--num_train_epochs",
                    default=3,
                    type=int)

parser.add_argument("--few_shot",
                    default=1000,
                    type=int)

# 200 (100) 400 (1K) 800 (10K)
parser.add_argument("--warmup_steps",
                    default=400,
                    type=int)

parser.add_argument("--ordering",
                    default="top")

parser.add_argument("--seed",
                    default=42,
                    type=int)

parser.add_argument("--gradient_accumulation_steps",
                    default=1,
                    type=int)

parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float)

parser.add_argument("-i",
                    "--input")

parser.add_argument("-o",
                    "--output")

args = parser.parse_args()

print("Using model: {}".format(args.model))
print("Input file: {}".format(args.input))
print("Output model: {}".format(args.output))
outdir = args.output

dataset = load_dataset("json",
                       data_files=args.input)

args.few_shot = min(len(dataset['train']), args.few_shot)

ids = range(args.few_shot)
if args.ordering == "bottom":
    ids = reversed(range(len(dataset['train'])-args.few_shot, len(dataset['train'])))
elif args.ordering == "random":
    dataset = dataset.shuffle(seed=args.seed)

dataset['train'] = dataset['train'].select(ids)
print("Few-shot: {}".format(args.few_shot))

def join_titles_and_abstracts(dataset, special_token="<s>"):
    dataset["src"] = "{}<s>{}".format(dataset["title"], dataset["abstract"])
    return dataset

dataset = dataset.map(join_titles_and_abstracts)

def join_phrases(dataset, special_token=";"):
    dataset["tgt"] = special_token.join(dataset["keyphrases"])
    return dataset

dataset = dataset.map(join_phrases)
    
dataset = dataset.remove_columns(["title", "abstract", "keyphrases"])

# create a tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model,
                                          token=args.token)
def preprocess_function(dataset):
    
    model_inputs = tokenizer(dataset["src"],
                             max_length=512,
                             truncation=True,
                             padding=True)
    
    labels = tokenizer(dataset["tgt"],
                       max_length=128,
                       truncation=True,
                       padding=True)

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets.set_format("torch")

model = BartForConditionalGeneration.from_pretrained(args.model,
                                                     token=args.token)
args = TrainingArguments(
    output_dir=outdir,
    #overwrite_output_dir=True,
    # take parameters from Meng 2023 paper
    learning_rate=args.learning_rate,
    #max_steps=max_steps,
    warmup_steps=args.warmup_steps,
    optim="adamw_torch",
    #adam_beta1=0.9,
    #adam_beta2=0.999,
    #adam_epsilon=1e-8,
    #weight_decay=0.01,
    #max_grad_norm=0.1,
    do_train=True,
    per_device_train_batch_size=args.batch_size,
    #per_device_eval_batch_size=batch_size,
    #save_total_limit=10,
    num_train_epochs=args.num_train_epochs,
    #max_steps=max_steps,
    #logging_steps=logging_steps,
    push_to_hub=False,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    #evaluation_strategy="epoch", #steps
    #load_best_model_at_end=True,
    #save_strategy="epoch"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,#NLPDataCollator(),
    train_dataset=tokenized_datasets["train"],
    #eval_dataset=tokenized_datasets["test"],
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]    
)

result = trainer.train()

trainer.save_model(output_dir=outdir)
tokenizer.save_pretrained(save_directory=outdir)

print(result)

