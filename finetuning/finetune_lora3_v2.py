# -*- coding: utf-8 -*-
"""
finetune_lora3_v2.py
=================
Description: Fine-tune MPT 7B on custom subtitles dataset. 
Using SFTTrainer from transformers library with QLoRA (4-bit quantization).
With adjusted learning rate, changed optimizer, set max gradient norm to 0.3 and set 
gradient eval steps to 8.

Author: Martin Tomasovic

Required libraries:
- peft
- trl
- torch
- pickle
- transformers
- datasets
- accelerate
- einops
- bitsandbytes
- loralib

Install the required libraries:
pip install peft
pip install trl
pip install -U bitsandbytes
pip install accelerate>=0.20.3 transformers
pip install datasets
pip install accelerate
pip install einops
pip install pickle
pip install torch
pip install loralib

Examples how to run:
run on metacentrum, by running script jobTrain2.sh, jobTrain3.sh or jobTrain4.sh and run_training.sh
"""

from datasets import load_dataset, load_from_disk
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from trl import SFTTrainer

import torch
import re
import pickle

import loralib as lora
from peft import LoraConfig, get_peft_model

import os

import argparse

parser=argparse.ArgumentParser()
parser.add_argument("modelSavePth")
parser.add_argument("r", type=int, default=8)
parser.add_argument("alpha", type=int, default=16)
parser.add_argument("dropout", type=float, default=0.1)

args=parser.parse_args()

SERVER_PTH = '/storage/brno2/home/martom198' # or '/storage/brno12-cerit/home/'

# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

#MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 1 
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 1e-6 # v2 = 5e-6 
LR_WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

ac = "place token here"

from huggingface_hub import login

login(token = ac)

# load model and tokenizer
name = "mistralai/Mistral-7B-Instruct-v0.3" #"mistralai/Mistral-7B-v0.3" #"google/gemma-3-12b-pt" #"mistralai/Mistral-7B-v0.3" #"tiiuae/falcon-7b"
model_pth = SERVER_PTH + "/lora/mpt-7b"

# Configure 4-bit quantization

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
"""
# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    load_in_4bit=False  # just to be explicit
)
"""

config = AutoConfig.from_pretrained(
    name,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    name,
    config=config,  # <--- Pass the modified config here
    quantization_config=bnb_config,  # Add 4-bit or 8-bit quantization
    trust_remote_code=True,
    device_map="auto",
    token=ac
)

def create_prompt(sample):
    bos_token = "<s>"
    system_message = 'Provide information about the movie in following format: {"rating": <rating>, "genres": <genres>, "review": <review_text>}.'
    eos_token = "</s>"
    ### Instruction: Provide information about the movie in following format: {\"rating\": <rating>, \"genres\": <genres>, \"review\": <review_text>}. \n\n### Input:
    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input:"
    full_prompt += "\n" + sample["prompt"]
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + sample["completion"]
    full_prompt += eos_token

    return full_prompt

tokenizer = AutoTokenizer.from_pretrained(name, max_length=32768)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

# set gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trained_model_save_path = SERVER_PTH + "/lora/knn_models/" + args.modelSavePth # path to store the fine-tuned adapters

"""Prepare the dataset"""
path_validation = SERVER_PTH + '/lora/dataset/rev3_validation_32.json'
path_train = SERVER_PTH + '/lora/dataset/rev3_train_32.json'
path_test = SERVER_PTH + '/lora/dataset/rev3_test_32.json'

# set the splits
data_files_validation = {
    "validation": path_validation,
}
data_files_train = {
    "train": path_train,
}
data_files_test = {
    "test": path_test,
}

# actually load
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=32768,
        return_tensors="pt"
    )

# get test dataset
train_dataset = load_from_disk(SERVER_PTH+'/lora/dataset/mistral32/m32_train_tokenized_v3')
valid_dataset = load_from_disk(SERVER_PTH+'/lora/dataset/mistral32/m32_validation_tokenized_v3')
train_dataset.set_format("torch", ["input_ids", "attention_mask"])
valid_dataset.set_format("torch", ["input_ids", "attention_mask"])

# prepare LoRA configuration
peft_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.alpha,
    lora_dropout=args.dropout,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Layers to apply LoRA
    task_type="CAUSAL_LM",
)

lora_alpha = 128 #16
lora_dropout = 0.05 #0.1
lora_rank = 64 #64

print(f"before: {sum(params.numel() for params in model.parameters() if params.requires_grad)}")
model = get_peft_model(model, peft_lora_config)
print(f"after: {sum(params.numel() for params in model.parameters() if params.requires_grad)}")

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

steps_per_epoch = int(len(train_dataset) / TRAIN_BATCH_SIZE)
eval_steps_count = round(steps_per_epoch / 3)

"""
the training arguments are inspired by: https://www.mldive.com/p/how-to-fine-tune-llama-2-with-lora
and https://rocm.blogs.amd.com/artificial-intelligence/llama2-lora/README.html
"""

# Set training arguments
training_args = TrainingArguments(
  output_dir= SERVER_PTH + '/lora/knn_training',
  overwrite_output_dir=True,
  num_train_epochs=1,
  gradient_accumulation_steps=8, # to update weights after more steps, necessery with batch size 1
  max_grad_norm=0.3,  # for stability with long sequences
  do_train=True,
  do_eval=True,
  per_device_train_batch_size=TRAIN_BATCH_SIZE,
  per_device_eval_batch_size=EVAL_BATCH_SIZE,
  warmup_steps=LR_WARMUP_STEPS,
  save_steps=eval_steps_count,
  save_total_limit=3,
  weight_decay=WEIGHT_DECAY,
  learning_rate=LEARNING_RATE,
  eval_strategy ='steps', # to evaluate every EVAL_STEPS_COUNT
  eval_steps=eval_steps_count,
  save_strategy='steps',
  load_best_model_at_end=True,
  metric_for_best_model='loss',
  greater_is_better=False,
  seed=SEED_TRAIN,
  optim="adafactor",
  logging_steps=200,
)

# Train the model
trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_lora_config,
)

trainer.train() # run fine-tuning
trainer.save_model(trained_model_save_path) #save custom model
