#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file for calculating perplexity of the model's output

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

ac = "" # Replace with your actual token
login(token=ac)

def extract_model_turns(text):
    #with open(file_path, 'r', encoding='utf-8') as f:
    #    text = f.read()

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

def calculate_perplexity(model, tokenizer, text):
    try:
        encodings = tokenizer(text, return_tensors="pt")
        encodings = {key: val.to(model.device) for key, val in encodings.items()}

        max_length = tokenizer.model_max_length
        stride = 512
        seq_len = encodings['input_ids'].size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            target_len = end_loc - prev_end_loc

            input_ids = encodings['input_ids'][:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-target_len] = -100  # mask overlap

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    except Exception as e:
        print(f"Error calculating perplexity: {str(e)}")
        return None


def process_batch(model, tokenizer, batch, template, system_msg, base_model_name):
    perplexities = []
    
    for sample in batch:
        try:
            content = sample["content"]
            movie_id = sample["num"]
            
            # Prepare messages with system prompt
            messages = [
                system_msg,
                {"role": "user", "content": content}
            ]

            # Generate prompt using template
            prompt = template.render(
                messages=messages,
                add_generation_prompt=True
            )
            
            # Generate response from the model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True
                )
            
            # Get the generated response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if base_model_name == "mistralai/Mistral-7B-Instruct-v0.3":
                turns = extract_model_turns(response)
                json_response = clean_turns(turns)
            else:
                json_start = response.find('"{') + 1
                json_end = response.rfind('}"') + 1
                json_response = response[json_start:json_end]

                json_objects = re.findall(r'\{.*?\}', json_response, re.DOTALL)
                if json_objects:
                    json_response = json_objects[0]
                else:
                    json_response = ""
            

            # Extract only the model's response (after the last assistant message)
            try:
                # Calculate perplexity only on the model's response
                ppl = calculate_perplexity(model, tokenizer, json_response)
                if ppl is not None:
                    perplexities.append({
                        "num": movie_id,
                        "perplexity": ppl,
                        "response": json_response
                    })
            except Exception as e:
                print(f"Error extracting model response for sample {movie_id}: {str(e)}")
                continue
            
        except Exception as e:
            print(f"Error processing sample {movie_id}: {str(e)}")
            continue
            
    return perplexities

def main():
    model_number = 11

     # Load model components
    base_model_name = "google/gemma-3-4b-pt"
    dataset_path = "/storage/brno2/home/xcerve30/test/datasets/test_dataset.json"
    template_path = "/storage/brno2/home/xcerve30/test/chat_template.json"

    if model_number == 1:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_8_16_0-1"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_8_model_perplexities.json"
    elif model_number == 2:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_32r_64a"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_32_model_perplexities.json"
    elif model_number == 3:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_64_128_0-1"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_64_model_perplexities.json"
    elif model_number == 4:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_256_512_0-1v2"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_256_model_perplexities_v2.json"
    elif model_number == 5:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_02_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_02_32_model_perplexities.json"
    elif model_number == 6:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_05_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_05_32_model_perplexities.json"
    elif model_number == 7:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_q_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_q_32_model_perplexities.json"
    elif model_number == 8:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_16"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_16_model_perplexities.json"
    elif model_number == 9:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_32_16"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_32_16_model_perplexities.json"
    elif model_number == 10:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_32_32"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_32_32_model_perplexities.json"
    elif model_number == 11:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/gemma_8"
        output_path = "/storage/brno2/home/xcerve30/test/results/gemma_8_2_model_perplexities.json"
    elif model_number == 12:
        peft_model_path = "/storage/brno2/home/xcerve30/test/models/mistral"
        output_path = "/storage/brno2/home/xcerve30/test/results/mistral_model_perplexities.json"
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    #dataset = dataset[:100]
    
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
    all_perplexities = []

    for i in range(0, len(dataset), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
        batch = dataset[i:i + batch_size]
        
        try:
            perplexities = process_batch(model, tokenizer, batch, template, system_msg, base_model_name)
            all_perplexities.extend(perplexities)
            
            # Save intermediate results after each batch
            with open(output_path, "w") as f:
                json.dump(all_perplexities, f, indent=2, ensure_ascii=False)

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

    # Calculate and print average perplexity
    if all_perplexities:
        avg_perplexity = sum(p["perplexity"] for p in all_perplexities) / len(all_perplexities)
        print(f"Average perplexity: {avg_perplexity:.2f}")
        perp_output_path = output_path.replace('.json', '_avg_perplexity.json')
        with open(perp_output_path, "w") as f:
            json.dump({"average_perplexity": avg_perplexity}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()