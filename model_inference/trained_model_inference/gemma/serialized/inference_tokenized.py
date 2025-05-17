#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file for running inference with the Gemma model with a pretokenized dataset

import sys
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from huggingface_hub import login
import gc
import os
import time
import traceback

# Disable the memory‐efficient and flash SDPA kernels, enable the safe math implementation
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

ac = "" # Replace with your actual token
login(token=ac)

# Set CUDA launch blocking for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def process_batch(model, tokenizer, batch):
    results = []
    results2 = []
    
    for sample in batch:
        try:
            start_time = time.time()
            movie_id = sample["movie_id"]
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=400,  # Set to 400 for balanced length
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    use_cache=True,  # Enable KV cache
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode and clean output
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            json_start = response.find('"{') + 1
            json_end = response.rfind('}"') + 1
            json_response = response[json_start:json_end]

            json_objects = re.findall(r'\{.*?\}', json_response, re.DOTALL)
            if json_objects:
                json_response = json_objects[0]
            else:
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
            
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Sample {movie_id} processed in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing sample {movie_id}: {str(e)}")
            print("Full error traceback:")
            print(traceback.format_exc())
            results.append("")  # Add empty result for failed sample
            results2.append("")  # Add empty result for failed sample
            continue
            
    return results, results2

def main():
    
    model_number = 11
    part = 10

    if model_number == 1:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_8_16_0-1"
        output_path = "/storage/brno2/home/xcerve30/test/results/5gemma_8_model_outputs.json"
    elif model_number == 2:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_32r_64a"
        output_path = "/storage/brno2/home/xcerve30/test/results/5gemma_32_model_outputs.json"
    elif model_number == 3:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_64_128_0-1"
        output_path = "/storage/brno2/home/xcerve30/test/results/5gemma_64_model_outputs.json"
    elif model_number == 4:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_256_512_0-1v2"
        output_path = "/storage/brno2/home/xcerve30/test/results/5gemma_256_model_outputs.json"
    elif model_number == 5:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_02_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/5gemma_02_32_model_outputs.json"
    elif model_number == 6:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_05_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/5gemma_05_32_model_outputs.json"
    elif model_number == 7:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_q_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/6gemma_q_32_model_outputs.json"
    elif model_number == 8:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_16"
        output_path = "/storage/brno2/home/xcerve30/test/results/6gemma_16_model_outputs.json"
    elif model_number == 9:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_32_16"
        output_path = "/storage/brno2/home/xcerve30/test/results/6gemma_32_16_model_outputs.json"
    elif model_number == 10:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_32_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/6gemma_32_32_model_outputs.json"
    elif model_number == 11:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_8"
        output_path = "/storage/brno2/home/xcerve30/test/results/9gemma_8_2_model_outputs.json"

    base_model_name = "google/gemma-3-4b-pt"
        
    # Load model components
    tokenized_dataset_path = "/storage/brno2/home/xcerve30/test/datasets/pretokenized_data/pretokenized_dataset.json"

    # Load tokenized dataset
    with open(tokenized_dataset_path, "r") as f:
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
        
        

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()

    # Enable model optimizations
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Process in smaller batches
    batch_size = 10  # Reduced batch size to 1 to prevent memory issues
    all_results = []
    all_results2 = []

    total_start_time = time.time()
    
    for i in range(0, len(dataset), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
        batch = dataset[i:i + batch_size]
        
        try:
            results, results2 = process_batch(model, tokenizer, batch)
            
            all_results.extend(results)
            all_results2.extend(results2)
            
            # Save intermediate results after each batch
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            print("Full error traceback:")
            print(traceback.format_exc())
            continue

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(dataset):.2f} seconds")

if __name__ == "__main__":
    main() 