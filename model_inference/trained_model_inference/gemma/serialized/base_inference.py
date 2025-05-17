#!/usr/bin/env python3

# file for running inference with the base Gemma model

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from huggingface_hub import login

# Disable the memory‐efficient and flash SDPA kernels, enable the safe math implementation
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def main():
    # Authenticate with HF Hub
    hf_token = os.getenv("HF_TOKEN", "")  # replace or set HF_TOKEN
    login(token=hf_token)

    model_name = "google/gemma-3-4b-it"
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Hard-coded prompt with JSON format instruction
    raw_prompt = (
        "You are a JSON-only assistant. Only output valid JSON without any extra text.\n"
        "Imagine a movie write a short review and the genre(s) of the movie."
        "Output exactly one valid JSON object with these keys:\n"
        "- review (string): your review of the subtitle\n"
        "- genres (array of strings): your guessed genres\n"
        "- rating (decimal number between 0 and 10)\n"
    )

    # Option A: use a text-generation pipeline directly
    try:
        chat_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        # Generate directly from raw_prompt
        result = chat_pipe(
            raw_prompt,
            max_length=200,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        # Print full generated text
        print("\n[Option A - Pipeline] Gemma says:\n", result[0]["generated_text"])
    except Exception as e:
        print(f"Option A (pipeline) failed: {e}")

    # Option B: manual generate() call
    try:
        inputs = tokenizer(raw_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
        # Decode and print full output
        full_resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n[Option B - Manual] Gemma says:\n", full_resp)
    except Exception as e:
        print(f"Option B (manual generate) failed: {e}")

if __name__ == "__main__":
    main()
