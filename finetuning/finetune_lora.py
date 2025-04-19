# -*- coding: utf-8 -*-
"""
finetune_lora.py
=================
Description: Fine-tune MPT 7B on custom subtitles dataset. 
Using SFTTrainer from transformers library and custom loss function.

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

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

import torch
import re
import pickle

import loralib as lora
from peft import LoraConfig, get_peft_model

import argparse

parser=argparse.ArgumentParser()
parser.add_argument("modelSavePth")
parser.add_argument("r", type=int, default=8)
parser.add_argument("alpha", type=int, default=16)
parser.add_argument("dropout", type=float, default=0.1)

args=parser.parse_args()

# HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

#MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 32 
EVAL_BATCH_SIZE = 32 
LEARNING_RATE = 5e-4 
LR_WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

# load model and tokenizer
name = "mosaicml/mpt-7b"

model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
        trust_remote_code=True,
        device_map="auto",
        return_dict_in_generate=True
    )

model.gradient_checkpointing_enable() # saves memory for longer sequences, prolongs computation a little bit

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token # add pad token

model.resize_token_embeddings(len(tokenizer)) # edit model size according to the new tokenizer size

# set gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

"""Prepare the dataset"""
# text dataset paths
train_text_path = '/storage/brno2/home/martom198/lora/XLgstext_train.txt'
validation_text_path = '/storage/brno2/home/martom198/lora/XLgstext_valid.txt'
# object dataset paths
train_dataset_path = "/storage/brno2/home/martom198/lora/train_dataset.pkl"
validation_dataset_path = "/storage/brno2/home/martom198/lora/valid_dataset.pkl"

trained_model_save_path = "/storage/brno2/home/martom198/lora/models/" + args.modelSavePth # path to store the fine-tuned adapters

def create_dataset(path):
    """
    function to create text dataset for fine-tuning
    """
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path= path,
        block_size=512
    )

# process training dataset
train_dataset = create_dataset(train_text_path)
# process validation dataset
valid_dataset = create_dataset(validation_text_path)

# store the train_dataset object to a file
with open(train_dataset_path, "wb") as f:
    pickle.dump(train_dataset, f)

# store the train_dataset object to a file
with open(validation_dataset_path, "wb") as f:
    pickle.dump(valid_dataset, f)

# load stored dataset
with open(train_dataset_path, "rb") as f:
    train_dataset = pickle.load(f)
with open(validation_dataset_path, "rb") as f:
    valid_dataset = pickle.load(f)


# prepare LoRA configuration
peft_lora_config = LoraConfig(
    r=args.r,
    lora_alpha=args.alpha,
    lora_dropout=args.dropout,
    task_type="CAUSAL_LM"
)
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
  output_dir='/storage/brno2/home/martom198/lora/training',
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
  evaluation_strategy='steps', # to evaluate every EVAL_STEPS_COUNT
  eval_steps=eval_steps_count,
  save_strategy='steps',
  load_best_model_at_end=True,
  metric_for_best_model='loss',
  greater_is_better=False,
  seed=SEED_TRAIN
)

# custom loss function
# adjusted code for our purpose from: https://discuss.huggingface.co/t/supervised-fine-tuning-trainer-custom-loss-function/52717
class SFTTrainerCustom(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(SFTTrainerCustom, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        
       # get label and prediction tokens
        outputs = model(**inputs)
        labels = inputs.get("labels")
        predicts = outputs.get("logits")
    
        # decode predictions and labels
        # Select token IDs based on softmax probabilities
        predicted_token_ids = softmax_selection(predicts, temperature=self.temperature)
        decoded_predictions = [tokenizer.decode(p.tolist()) for p in predicted_token_ids]
        decoded_labels = [tokenizer.decode(l.tolist()) for l in labels]

        # function to output quantities to a list       
        predicted_quantities = [extract_score_and_genres(p) for p in decoded_predictions]
        actual_quantities = [extract_score_and_genres(l) for l in decoded_labels]
        predicted_quantities, actual_quantities = torch.quantities(decoded_predictions, decoded_labels)
        
        predicted_tensor = torch.tensor(predicted_quantities, device=model.device)
        actual_tensor = torch.tensor(actual_quantities, device=model.device)
        predicted_tensor.requires_grad_()
        
        # Compute MSE loss
        loss_function = torch.nn.MSELoss()
        loss = loss_function(predicted_tensor, actual_tensor)
        
        return (loss, outputs) if return_outputs else loss

def softmax_selection(predictions, temperature=1.0):
        """
        Apply softmax to model predictions and sample a token based on the resulting probabilities.

        Args:
            predictions (torch.Tensor): The tensor containing the raw predictions from the model.
            temperature (float): Temperature parameter to adjust the sharpness of the probability distribution.
                                A lower temperature makes the distribution sharper.

        Returns:
            torch.Tensor: Tensor containing the selected token IDs.
        """
        # Apply softmax with temperature
        probs = torch.F.softmax(predictions / temperature, dim=-1)

        # Sampling a token based on the probabilities
        sampled_tokens = torch.multinomial(probs, num_samples=1)

        return sampled_tokens

def extract_score(text):
    # extract a score and genre tokens
    # example input: "Review: ... Score: 7.5 Genres: Drama, Thriller, Mystery"
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text)
    score = float(match.group(1)) if match else 0.0
    return score

try:
    # Train the model
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    peft_config=peft_lora_config,
    dataset_text_field="text" # dictates dataset formatting
    )
except Exception as e:
    print(str(e))

trainer.train() # run fine-tuning
trainer.save_model(trained_model_save_path) #save custom model
