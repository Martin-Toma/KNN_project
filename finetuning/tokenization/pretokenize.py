# -*- coding: utf-8 -*-


from datasets import load_dataset
from transformers import AutoTokenizer
from jinja2 import Environment, BaseLoader
import json
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
SERVER_PTH = "."  # '/storage/brno12-cerit/home/martom198'

ac = "FILLMEIN"

from huggingface_hub import login

login(token=ac)

with open("chat_template.json", "r") as f:
    raw = json.load(f)
chat_template = raw["chat_template"]
jinja_env = Environment(
    loader=BaseLoader(),
    keep_trailing_newline=True,     # preserve final newline
)
template = jinja_env.from_string(chat_template)
# 2) Load tokenizer
name = "google/gemma-3-4b-pt"
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token

# 3) Prepare paths
path_train = "dataset/rev3_train_32.json"
path_validation = "dataset/rev3_validation_32.json"
path_test = "dataset/rev3_test_32.json"

# Tokenize function, batched
def tokenize(batch):
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    max_length = 32000

    for prompt, completion in zip(batch["prompt"], batch["completion"]):
        # 2a) Render the prefix up through '<start_of_turn>model\n'
        while True:
                
            prefix = template.render(
                messages=[system_msg, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )

            # 2b) Encode prefix (no special tokens) & completion (+ eos)
            prefix_ids     = tokenizer.encode(prefix,            add_special_tokens=False)
            completion_ids = tokenizer.encode(completion,        add_special_tokens=False)
            completion_ids = completion_ids + [tokenizer.eos_token_id]

            # 2c) Build input_ids and labels
            input_ids = prefix_ids + completion_ids
            labels    = [-100] * len(prefix_ids) + completion_ids.copy()

            # 2d) Truncate from the left if too long
            if len(input_ids) > max_length:
                diff = len(input_ids) - max_length
                prompt = prompt[:len(prompt)-diff]
                continue

            # 2e) Build attention mask & pad out to max_length
            attention_mask = [1] * len(input_ids)
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids      += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels         += [-100] * pad_len
            break
        # 2f) Collect
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids":      input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels":         labels_batch,
    }

# 3) Load & tokenize your JSON dataset
dataset = load_dataset(
    "json",
    data_files={
        "train":      path_train,
        "validation": path_validation,
        "test":       path_test,
    },
)
tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["prompt", "completion"],
    desc="Tokenizing dataset with chat template",
)

# set PyTorch format just in case
tokenized.set_format("torch", ["input_ids", "attention_mask", "labels"])

# Save to disk
tokenized["train"].save_to_disk("g32_train_tokenized")
tokenized["validation"].save_to_disk("g32_validation_tokenized")
