#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import torch
from jinja2 import Environment, BaseLoader
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def main():
    if len(sys.argv) != 2:
        print("Usage: inference.py <input_file.txt>")
        sys.exit(1)

    # Load configuration
    with open("chat_template.json", "r") as f:
        chat_template = json.load(f)["chat_template"]
    
    # Load model components
    base_model_name = "google/gemma-3-4b-pt"
    peft_model_path = "./lora/knn_models/gemma_8_16_0-1"  # Update with your path
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()

    # Read input file
    with open(sys.argv[1], "r") as f:
        txt = f.read()                           
        # Step 1: remove indexes, timestamps, tags
        txt = re.sub(                                                           
            r'(?m)(?:^[0-9]+\r?\n|^\d{2}:\d{2}:\d{2},\d{3} -->.*\r?\n|<[^>]+>)',
            "",
            txt,
        )
                                      
        # Step 2: collapse blank lines
        txt = re.sub(r'(?m)^\s*\r?\n', "", txt)

        subtitle = txt.strip()
        txt=None

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
            "Ensure the output is parseable JSON."
        )
    }
    
    messages = [
        system_msg,
        {"role": "user", "content": subtitle}
    ]

    # Generate prompt
    prompt = template.render(
        messages=messages,
        add_generation_prompt=True
    )

    # Tokenize and generate
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

    # Decode and clean output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    json_response = response[json_start:json_end]

    try:
        parsed = json.loads(json_response)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("Error: Model generated invalid JSON")
        print("Raw output:", response)

if __name__ == "__main__":
    main()
