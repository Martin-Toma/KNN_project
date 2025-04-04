"""
Estimates the number of tokens per movie subtitle

Autor: M. Tomasovic
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = "tiiuae/Falcon3-3B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_name)

fName = r"C:\Users\marti\Music\knn\KNN_project\row_output.txt"
fNameii = r"C:\Users\marti\Downloads\The.Electric.State.2025.1080p.NF.WEB-DL.DDP5.1.Atmos.H.264-FLUX.en.cc.srt"
fNamei = r"C:\Users\marti\Downloads\Invincible S03E08 cz tit 1080p by PEFTO.srt"

with open(fName, "r") as file:

    prompt = file.read()

    messages = [
            {"role": "system", "content": "You are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible."},
            {"role": "user", "content": prompt}
        ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt")

    print(model_inputs)
    print("Number of tokens in tokenized prompt: ", (model_inputs['input_ids'].shape)[1])
