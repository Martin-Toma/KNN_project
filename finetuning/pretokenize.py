# -*- coding: utf-8 -*-
"""
finetune_lora.py
=================
Description: Fine-tune MPT 7B on custom subtitles dataset. 
Using SFTTrainer from transformers library.

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

from datasets import load_dataset
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig

import torch
import re
import pickle

SERVER_PTH = '/storage/brno2/home/martom198' #'/storage/brno12-cerit/home/martom198'

ac = "put here your token"

from huggingface_hub import login

login(token = ac)

# load model and tokenizer
name = "google/gemma-3-12b-pt" #"mistralai/Mistral-7B-v0.3" #"tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token # add pad token

#model.resize_token_embeddings(len(tokenizer)) # edit model size according to the new tokenizer size

# set gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""Prepare the dataset"""
# load dataset from the HuggingFace Hub
#dataset = load_dataset("Nirmata/Movie_evaluation")

# object dataset paths
train_dataset_path = "/storage/brno2/home/martom198/lora/dataset/train_dataset.pkl"
validation_dataset_path = "/storage/brno2/home/martom198/lora/dataset/valid_dataset.pkl"
test_dataset_path = "/storage/brno2/home/martom198/lora/dataset/test_dataset.pkl"

path_validation = SERVER_PTH + '/lora/dataset/validation_trimmed.json'
path_train = SERVER_PTH + '/lora/dataset/train_trimmed.json'
path_test = SERVER_PTH + '/lora/dataset/test_trimmed.json'

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

# get training dataset
train_dataset = load_dataset("json", data_files=data_files_train)["train"]
# get validation dataset
valid_dataset = load_dataset("json", data_files=data_files_validation)["validation"]
# get test dataset
test_dataset = load_dataset("json", data_files=data_files_test)["test"]

# store the train_dataset object to a file
with open(train_dataset_path, "wb") as f:
    pickle.dump(train_dataset, f)

# store the validation_dataset object to a file
with open(validation_dataset_path, "wb") as f:
    pickle.dump(valid_dataset, f)

# store the test_dataset object to a file
with open(test_dataset_path, "wb") as f:
    pickle.dump(test_dataset, f)

print("All loaded and saved")