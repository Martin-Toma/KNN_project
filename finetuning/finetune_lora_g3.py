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

from datasets import load_dataset, load_from_disk
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from bitsandbytes.optim import Adam8bit
from peft import prepare_model_for_kbit_training 
import torch
import re
import pickle

import loralib as lora
from peft import LoraConfig, get_peft_model
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import argparse

parser=argparse.ArgumentParser()
parser.add_argument("modelSavePth")
parser.add_argument("r", type=int, default=8)
parser.add_argument("alpha", type=int, default=16)
parser.add_argument("dropout", type=float, default=0.05)

args=parser.parse_args()

SERVER_PTH = '/storage/brno2/home/martom198' #'/storage/brno12-cerit/home/martom198'

# HYPERPARAMS
SEED_SPLIT =42
SEED_TRAIN = 42

#MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 5e-4 
LR_WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

ac = "place hf token here"

from huggingface_hub import login

login(token = ac)


# load model and tokenizer
name = "google/gemma-3-4b-pt" #"google/gemma-3-12b-pt" #"mistralai/Mistral-7B-v0.3" #"tiiuae/falcon-7b"


# set attention implementation to "torch"
config = AutoConfig.from_pretrained(
    name,
    trust_remote_code=True,
    #attn_implementation='eager'
    attn_config={
      "attn_impl": "flash",
      "use_flash_attn": True,
      "attn_uses_sequence_id": False
    }
)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                 # ← quantize all weights to 8-bit
    bnb_8bit_compute_dtype=torch.float16  # run computations in fp16
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    config=config,  # <--- Pass the modified config here
    #torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quant_config,
    token=ac
    #force_download=True,
    #resume_download=False
)
model = prepare_model_for_kbit_training(model) 
model.config.use_cache = False
model.gradient_checkpointing_enable()
# is not support model.gradient_checkpointing_enable() # saves memory for longer sequences, prolongs computation a little bit
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

trained_model_save_path = SERVER_PTH + "/lora/knn_models/" + args.modelSavePth # path to store the fine-tuned adapters

"""Prepare the dataset"""

path_validation = SERVER_PTH + 'dataset/validation_trimmed.json'
path_train = SERVER_PTH + 'dataset/train_trimmed.json'
path_test = SERVER_PTH + 'dataset/test_trimmed.json'

# actually load

# get training dataset
train_d_path = "/storage/brno2/home/martom198/lora/dataset/gemma32/g32_train_tokenized"
valid_d_path = "/storage/brno2/home/martom198/lora/dataset/gemma32/g32_validation_tokenized"
train_dataset = load_from_disk(train_d_path)
valid_dataset = load_from_disk(valid_d_path)
train_dataset.set_format("torch", ["input_ids", "attention_mask","labels"])
valid_dataset.set_format("torch", ["input_ids", "attention_mask","labels"])

# prepare LoRA configuration
peft_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.alpha,
    lora_dropout=args.dropout,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Layers to apply LoRA
    task_type="CAUSAL_LM",
)


print(f"base trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
model = get_peft_model(model, peft_lora_config)
print(f"LoRA   trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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
  save_total_limit=4,
  weight_decay=WEIGHT_DECAY,
  learning_rate=LEARNING_RATE,
  eval_strategy ='steps', # to evaluate every EVAL_STEPS_COUNT
  eval_steps=eval_steps_count,
  save_strategy='steps',
  fp16=True,
  gradient_accumulation_steps=8,
  load_best_model_at_end=True,
  metric_for_best_model='loss',
  greater_is_better=False,
  label_names=["labels"],
  seed=SEED_TRAIN,
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

# Tell Trainer which dataset column holds the labels:
model.config.label_names = ["labels"]
# And also for the underlying base model (PEFT sometimes hides this setting):
if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
    model.base_model.config.label_names = ["labels"]

# (Optional) sanity‐check:
print(" label_names:", model.config.label_names,
      "base label_names:", getattr(model.base_model.config, "label_names", None))

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
