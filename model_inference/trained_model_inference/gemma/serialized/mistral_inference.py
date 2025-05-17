#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import torch
from jinja2 import Environment, BaseLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from huggingface_hub import login
import math
import gc
import os

# Disable the memory‐efficient and flash SDPA kernels, enable the safe math implementation
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# Set CUDA launch blocking for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def extract_model_turns(text):
    # Regex to match from <start_of_turn>model to </start_of_turn> or <end_of_turn>
    pattern = re.compile(
        r'(<start_of_turn>model.*?)(</start_of_turn>|<end_of_turn>|})',
        re.DOTALL
    )
    matches = pattern.findall(text)
    # Extract only the matched text (group 1 + group 2)
    # Join all matched text segments into a single string
    return (matches[0][0] + matches[0][1]) if matches else ''

def clean_turns(turn):
    turn = re.sub(r'^.*?{', '{', turn, flags=re.DOTALL)
    turn = re.sub(r'}[^}]*$', '}', turn, flags=re.DOTALL)
    turn = turn.replace('\n', '')
    if turn.startswith('<start_of_turn>model'):
        turn = '{' + turn[len('<start_of_turn>model'):].rstrip('}') + '}'
    turn = turn.replace('</start_of_turn>', '')
    turn = turn.replace('<end_of_turn>', '')
    return turn.strip()

def process_batch(model, tokenizer, batch, template, system_msg, base_model_name):
    results = []
    results2 = []
    perplexities = []
    
    for sample in batch:
        response = ""  # Initialize response variable
        try:
            content = sample["content"]
            movie_id = sample["num"]

            # Prepare messages
            messages = [
                system_msg,
                {"role": "user", "content": content}
            ]

            # Generate prompt
            prompt = template.render(
                messages=messages,
                add_generation_prompt=True
            )

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Clear CUDA cache before generation
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True
                )

            # Decode and clean output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if base_model_name == "mistralai/Mistral-7B-Instruct-v0.3":
                try:
                    turns = extract_model_turns(response)
                    json_response = clean_turns(turns)
                except Exception as e:
                    print(f"Error processing sample {movie_id}: {str(e)}")
                    json_response = ""

            if json_response:
                try:
                    parsed = json.loads(json_response)
                    parsed["num"] = movie_id
                    json_response = json.dumps(parsed)
                except json.JSONDecodeError:
                    pass
                
            results.append(json_response)
            results2.append(response)
            
        except Exception as e:
            print(f"Error processing sample {movie_id}: {str(e)}")
            results2.append(response)  # Now response is always defined
            results.append("")  # Add empty result for failed sample
            continue
            
    return results, results2, perplexities

def main():
    ac = "" # Replace with your actual token
    login(token=ac)

    part = 1

    peft_model_path = "/storage/brno2/home/xcerve30/test/models/mistral"
    output_path = "/storage/brno2/home/xcerve30/test/results/9mistral_model_outputs.json"
        
    # Load model components
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    dataset_path = "/storage/brno2/home/xcerve30/test/datasets/test_dataset.json"
    template_path = "/storage/brno2/home/xcerve30/test/chat_template.json"

    # Load tokenized dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if part == 1:
        dataset = dataset[:100]
    elif part == 2:
        dataset = dataset[100:200]
    elif part == 3:
        dataset = dataset[200:300]
    elif part == 4:
        dataset = dataset[300:400]
    elif part == 5:
        dataset = dataset[400:500]
    elif part == 6:
        dataset = dataset[500:600]
    elif part == 7:
        dataset = dataset[600:700]
    elif part == 8:
        dataset = dataset[700:800]
    elif part == 9:
        dataset = dataset[800:900]
    elif part == 10:
        dataset = dataset[900:1000]
        


    # Load configuration
    with open(template_path, "r") as f:
        chat_template = json.load(f)["chat_template"]
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()

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

    # Process in smaller batches
    batch_size = 5
    all_results = []
    all_results2 = []
    all_perplexities = []

    for i in range(0, len(dataset), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
        batch = dataset[i:i + batch_size]
        
        try:
            results, results2, perplexities = process_batch(model, tokenizer, batch, template, system_msg, base_model_name)
            
            all_results.extend(results)
            all_results2.extend(results2)
            all_perplexities.extend(perplexities)
            
            # Save intermediate results after each batch
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            output_path2 = output_path.replace(".json", "_2.json")
            with open(output_path2, "w") as f:
                json.dump(all_results2, f, indent=2, ensure_ascii=False)

            try:
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error clearing GPU memory: {str(e)}")
                continue
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue

if __name__ == "__main__":
    main()