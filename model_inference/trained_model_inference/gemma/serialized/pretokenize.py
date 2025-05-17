#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file for pretokenizing the dataset for the Gemma model

import json
import torch
from transformers import AutoTokenizer
from jinja2 import Environment, BaseLoader
import os
from tqdm import tqdm
import numpy as np
from huggingface_hub import login
import traceback
import re

# Maximum number of tokens for input (leaving space for generation)
MAX_IN_TOKENS = 28000

def clean_subtitle(text: str) -> str:
    # remove indexes, timestamps, tags
    text = re.sub(
        r'(?m)(?:^[0-9]+\r?\n|^\d{2}:\d{2}:\d{2},\d{3} -->.*\r?\n|<[^>]+>)',
        "",
        text,
    )
    # collapse blank lines
    text = re.sub(r'(?m)^\s*\r?\n', "", text)
    text = text.strip()
    return text

def cut_middle(text, max_size, tokenizer):
    """Cut the middle portion of text to fit within max_size tokens"""
    first_half = max_size // 2
    second_half = max_size - first_half

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_size:
        return text  # no cut needed
    
    # cut middle portion
    tokens_trimmed = tokens[:first_half] + tokens[-second_half:]
    # decode back to text
    return tokenizer.decode(tokens_trimmed, skip_special_tokens=True)

def load_config():
    # Load configuration
    template_path = "/storage/brno2/home/xcerve30/test/chat_template.json" # TODO: change to the correct path
    dataset_path = "/storage/brno2/home/xcerve30/test/datasets/test_dataset.json"
    
    with open(template_path, "r") as f:
        chat_template = json.load(f)["chat_template"]
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    return chat_template, dataset

def prepare_messages(content, system_msg):
    return [
        system_msg,
        {"role": "user", "content": content}
    ]

def pretokenize_dataset():
    # Load model components
    base_model_name = "google/gemma-3-4b-pt"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load configuration
    chat_template, dataset = load_config()
    
    # Prepare chat template
    jinja_env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    template = jinja_env.from_string(chat_template)
    
    system_msg = {
        "role": "system",
        "content": (
            "You only respond with JSON and nothing else. "
            "When given a user prompt containing a subtitle, write a short review and guess the genre(s) based on that subtitle. "
            "Output exactly one valid JSON object with these keys:\n"
            "- review (string): your review of the subtitle\n"
            "- genres (array of strings): your guessed genres\n"
            "- rating (decimal number between 0 and 10)\n"
            "Do NOT include any additional keys, comments, or markdown. "
            "Ensure the output is ONLY ONE parseable JSON."
        )
    }
    
    # Create output directory if it doesn't exist
    output_dir = "/storage/brno2/home/xcerve30/test/datasets/pretokenized_data" # TODO: change to the correct path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pretokenized_dataset.json")
    
    # Process dataset
    tokenized_data = []
    trimmed_count = 0
    token_counts = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Pretokenizing dataset")):
        try:
            content = sample["content"]
            movie_id = sample["num"]

            content = clean_subtitle(content)
            
            # Prepare messages
            messages = prepare_messages(content, system_msg)
            
            # Generate prompt
            prompt = template.render(
                messages=messages,
                add_generation_prompt=True
            )
            
            # Check and trim if necessary
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            original_token_count = len(prompt_tokens)
            
            if original_token_count > MAX_IN_TOKENS:
                prompt = cut_middle(prompt, MAX_IN_TOKENS, tokenizer)
                trimmed_count += 1
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            
            # Tokenize using the same approach as 2inference.py
            tokenized = tokenizer(prompt, return_tensors="pt")
            
            final_token_count = len(tokenized["input_ids"][0])
            token_counts.append(final_token_count)
            
            # Print token counts for this sample
            #print(f"Sample {movie_id}: {original_token_count} -> {final_token_count} tokens")
            
            # Convert to list for JSON serialization
            tokenized_dict = {
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist(),
                "movie_id": movie_id,
                #"original_content": content,
                "was_trimmed": original_token_count > MAX_IN_TOKENS,
                "token_count": final_token_count
            }
            
            tokenized_data.append(tokenized_dict)
            
            # Save intermediate results every 100 samples
            if (i + 1) % 500 == 0:
                # Save final results
                with open(output_path, "w") as f:
                    json.dump(tokenized_data, f, indent=2)
            
        except Exception as e:
            print(f"Error processing sample {movie_id}: {str(e)}")
            print("Full error traceback:")
            print(traceback.format_exc())
            continue
    
    # Calculate and print statistics
    token_counts = np.array(token_counts)
    print("\nToken Count Statistics:")
    print(f"Total samples processed: {len(tokenized_data)}")
    print(f"Samples that needed trimming: {trimmed_count}")
    print(f"Average tokens per sample: {np.mean(token_counts):.2f}")
    print(f"Min tokens: {np.min(token_counts)}")
    print(f"Max tokens: {np.max(token_counts)}")
    print(f"Median tokens: {np.median(token_counts):.2f}")
    print(f"Standard deviation: {np.std(token_counts):.2f}")
    
    # Save final results
    with open(output_path, "w") as f:
        json.dump(tokenized_data, f, indent=2)
    
    print(f"\nFinal results saved to {output_path}")

if __name__ == "__main__":
    login(token="") # TODO: change to the correct token
    pretokenize_dataset() 