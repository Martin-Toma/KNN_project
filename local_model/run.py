"""
Runs inference of small 3B llm model, preparation for bigger one

Autor: M. Tomasovic
"""
# adjusted script original from: https://huggingface.co/tiiuae/Falcon3-3B-Instruct 
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = "tiiuae/Falcon3-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def botAnswer(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

try:
    with open(r"C:\Users\marti\Music\knn\separate project\prompts.json", "r") as pf:
            promptFile = json.load(pf)             # load data from prompts file
            for prompt in promptFile:              # iterate over prompts
                botAnswer(prompt)
except Exception as e:
    print(e)