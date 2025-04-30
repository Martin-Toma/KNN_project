# -*- coding: utf-8 -*-
"""
finetune_lora.py
=================
Description: Fine-tune MPT 7B on custom subtitles dataset. 
Using SFTTrainer from transformers library with QLoRA (4-bit quantization).

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

SERVER_PTH = '/storage/brno2/home/martom198' #'/storage/brno12-cerit/home/martom198'

# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

#MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 1 
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 5e-4 
LR_WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

ac = "here should be the token"

from huggingface_hub import login

login(token = ac)


# load model and tokenizer
name = "mistralai/Mistral-7B-v0.3" #"google/gemma-3-12b-pt" #"mistralai/Mistral-7B-v0.3" #"tiiuae/falcon-7b"
model_pth = SERVER_PTH + "/lora/mpt-7b"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# set attention implementation to "torch"
"""
config = AutoConfig.from_pretrained(
    name,
    trust_remote_code=True
)
#config.attn_config['attn_impl'] = 'flash' #'torch'
"""
"""
model = AutoModelForCausalLM.from_pretrained(
        name,
        #config=config,
        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
        trust_remote_code=True,
        device_map="auto",
        attn_impl = "torch",  # Use standard PyTorch attention (NOT Triton/Flash Attention)
        use_flash_attn = False,  # Redundant, but let's be explicit
        attn_uses_sequence_id = False
    )
"""
"""
model = AutoModelForCausalLM.from_pretrained(
        name,
        #config=config,
        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
        trust_remote_code=True,
        device_map="auto",
        force_download=True,
        resume_download=False,
        attn_impl = "torch",  # Use standard PyTorch attention (NOT Triton/Flas>
        use_flash_attn = False,  # Redundant, but let's be explicit
        attn_uses_sequence_id = False
    )
"""

config = AutoConfig.from_pretrained(
    name,
    trust_remote_code=True,
    #attn_config={
    #    "attn_impl": "torch",  # Use standard PyTorch attention (NOT Triton/Flash Attention)
    #    "use_flash_attn": False,  # Redundant, but let's be explicit
    #    "attn_uses_sequence_id": False  # Sometimes required for older MPT versions
    #}
)

model = AutoModelForCausalLM.from_pretrained(
    name,
    config=config,  # <--- Pass the modified config here
    quantization_config=bnb_config,  # Add 4-bit quantization
    trust_remote_code=True,
    #device_map="auto",
    token=ac
    #force_download=True,
    #resume_download=False
)

# is not support model.gradient_checkpointing_enable() # saves memory for longer sequences, prolongs computation a little bit

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer))
#tokenizer.add_special_tokens=False
#tokenizer.pad_token = tokenizer.eos_token # add pad token
#tokenizer.padding_side = "right"  # Important for consistent behavior
#tokenizer.add_eos_token = True 

#model.resize_token_embeddings(len(tokenizer)) # edit model size according to the new tokenizer size

# set gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
trained_model_save_path = SERVER_PTH + "/lora/knn_models/" + args.modelSavePth # path to store the fine-tuned adapters

"""Prepare the dataset"""
# load dataset from the HuggingFace Hub
#dataset = load_dataset("Nirmata/Movie_evaluation")
"""
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
"""
# actually load

# get training dataset
#train_dataset = load_dataset("json", data_files=data_files_train, streaming=True)["train"]
# get validation dataset
#valid_dataset = load_dataset("json", data_files=data_files_validation, streaming=True)["validation"]
# get test dataset
# test_dataset = load_dataset("json", data_files=data_files_test)["test"]
"""
# store the train_dataset object to a file
with open(train_dataset_path, "wb") as f:
    pickle.dump(train_dataset, f)

# store the validation_dataset object to a file
with open(validation_dataset_path, "wb") as f:
    pickle.dump(valid_dataset, f)

# store the test_dataset object to a file
with open(test_dataset_path, "wb") as f:
    pickle.dump(test_dataset, f)

# load stored dataset
with open(train_dataset_path, "rb") as f:
    train_dataset = pickle.load(f)
with open(validation_dataset_path, "rb") as f:
    valid_dataset = pickle.load(f)
"""

scratch_dir = os.getenv("SCRATCHDIR")
train_dataset = load_from_disk(scratch_dir+'/train_tokenized')
valid_dataset = load_from_disk(scratch_dir+'validation_tokenized')
train_dataset.set_format("torch", ["input_ids", "attention_mask"])
valid_dataset.set_format("torch", ["input_ids", "attention_mask"])

# prepare LoRA configuration
peft_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.alpha,
    lora_dropout=args.dropout,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Layers to apply LoRA
    task_type="CAUSAL_LM",
    #bias="none",
    #target_modules=[
    #    "query_key_value",
    #    "dense",
    #    "dense_h_to_4h",
    #    "dense_4h_to_h",
    #]
    #target_modules=["q_proj", "v_proj"],  # modules to adapt
)

lora_alpha = 8 #16
lora_dropout = 0.05 #0.1
lora_rank = 8 #64

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
  dataset_text_field="chat_sample",
  seed=SEED_TRAIN
  #max_length=1024
)

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

#try:
    # Train the model
trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    #formatting_func=create_prompt,
    #tokenizer=tokenizer,
    peft_config=peft_lora_config,
    #dataset_text_field="text" # dictates dataset formatting
)
#except Exception as e:
#    print(str(e))

trainer.train() # run fine-tuning
trainer.save_model(trained_model_save_path) #save custom model
