# -*- coding: utf-8 -*-
"""
test_tokenization.py
"""

from datasets import load_dataset, load_from_disk
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from transformers import BitsAndBytesConfig

import torch
import re
import pickle

import os

SERVER_PTH = '/storage/brno2/home/martom198' #'/storage/brno12-cerit/home/martom198'

# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

#MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 1 
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 1e-5 
LR_WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

ac = "place token here"

from huggingface_hub import login

login(token = ac)

# load model and tokenizer
name = "mistralai/Mistral-7B-Instruct-v0.3" #"mistralai/Mistral-7B-v0.3" #"google/gemma-3-12b-pt" #"mistralai/Mistral-7B-v0.3" #"tiiuae/falcon-7b"
model_pth = SERVER_PTH + "/lora/mpt-7b"



config = AutoConfig.from_pretrained(
    name,
    trust_remote_code=True,
    #attn_config={
    #    "attn_impl": "torch",  # Use standard PyTorch attention (NOT Triton/Flash Attention)
    #    "use_flash_attn": False,  # Redundant, but let's be explicit
    #    "attn_uses_sequence_id": False  # Sometimes required for older MPT versions
    #}
)

# is not support model.gradient_checkpointing_enable() # saves memory for longer sequences, prolongs computation a little bit
def create_prompt(sample):
    bos_token = "<s>"
    system_message = 'Provide information about the movie in following format: {"rating": <rating>, "genres": <genres>, "review": <review_text>}.'
    eos_token = "</s>"

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
tokenizer.add_special_tokens=False

# set gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Prepare the dataset"""
# load dataset from the HuggingFace Hub
#dataset = load_dataset("Nirmata/Movie_evaluation")

# object dataset paths
train_dataset_path = "/storage/brno2/home/martom198/lora/dataset/train_dataset.pkl"
validation_dataset_path = "/storage/brno2/home/martom198/lora/dataset/valid_dataset.pkl"
test_dataset_path = "/storage/brno2/home/martom198/lora/dataset/test_dataset.pkl"

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

#train_dataset = load_from_disk(SERVER_PTH+'/lora/dataset/mistral32/m32_train_tokenized_v3')
valid_dataset = load_from_disk(r"C:\Users\marti\Downloads\m32_validation_tokenized_v3")
#train_dataset.set_format("torch", ["input_ids", "attention_mask"])
valid_dataset.set_format("torch", ["input_ids", "attention_mask"])

sample_input_ids = valid_dataset[0]["input_ids"]
decoded_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
with open("test_decoded.txt", "w", encoding="utf-8") as d:
    d.write(decoded_text)
print("First decoded input:\n", decoded_text)