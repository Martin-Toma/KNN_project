# new_pretokenize.py
# -*- coding: utf-8 -*-
"""
Tokenize subtitles → inputs for multi-task Gemma3.

Produces and saves train/validation splits with:
- input_ids, attention_mask, labels        (for LM head)
- rating_label                             (float)
- genre_labels                             (multi-hot list)
"""
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from jinja2 import Environment, BaseLoader
from datasets import Dataset, DatasetDict


def load_json_file(path):
    # If your file is a single JSON array:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # this gives you a list of dicts

    # If it's JSONL (one JSON object per line), instead do:
    # with open(path) as f:
    #     return [json.loads(line) for line in f]


# 1) Load raw splits as Python lists of dicts
path_train = "dataset_splits/2instruction_train_dataset.json"
path_validation = "dataset_splits/2instruction_eval_dataset.json"
path_test = "dataset_splits/2instruction_test_dataset.json"

# 4) Bundle into a DatasetDict so you can still call .map()
raw = load_dataset(
    "json",
    data_files={
        "train": path_train,
        "validation": path_validation,
        "test": path_test,
    },
)

# 2) Build genre2id mapping from train
all_genres = set()
for ex in raw["train"]:
    parsed = json.loads(ex["completion"])
    if type(parsed["genres"]) != list:
        continue  # Empty genre
    all_genres.update(parsed["genres"])
genre_list = sorted(all_genres)
genre2id = {g: i for i, g in enumerate(genre_list)}
with open("genre2id.json", "w") as f:
    json.dump(genre2id, f)
num_genres = len(genre_list)

# 3) Prepare tokenizer + template
name = "google/gemma-3-4b-pt"
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token

with open("chat_template.json", "r") as f:
    chat = json.load(f)["chat_template"]
jinja_env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
template = jinja_env.from_string(chat)
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
    ),
}

# 4) Tokenize + extract labels
max_length = 8000


def tokenize(batch):
    in_ids, attn, lm_labels = [], [], []
    rating_lbls, genre_lbls = [], []

    for prompt, completion in zip(batch["prompt"], batch["completion"]):
        # parse JSON completion
        parsed = json.loads(completion)
        rating = float(parsed["rating"])
        genres = parsed["genres"]
        # build multi-hot
        gh = [0] * num_genres
        for g in genres:
            try:
                gh[genre2id[g]] = 1
            except KeyError:
                break

        # render prefix → full text to generate
        while True:
            prefix = template.render(
                messages=[system_msg, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            pref_ids = tokenizer.encode(prefix, add_special_tokens=False)
            comp_ids = tokenizer.encode(parsed["review"], add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            inp = pref_ids + comp_ids
            lbl = [-100] * len(pref_ids) + comp_ids.copy()

            # truncate if over
            if len(inp) > max_length:
                prompt = prompt[:max_length - (len(inp) - len(prompt))]
                continue

            # pad
            pad = max_length - len(inp)
            if pad > 0:
                inp += [tokenizer.pad_token_id] * pad
                lbl += [-100] * pad
            break

        in_ids.append(inp)
        attn.append([1] * (len(inp) - pad) + [0] * pad)
        lm_labels.append(lbl)
        rating_lbls.append(rating)
        genre_lbls.append(gh)

    return {
        "input_ids": in_ids,
        "attention_mask": attn,
        "labels": lm_labels,
        "rating_label": rating_lbls,
        "genre_labels": genre_lbls,
    }


# 5) Map + save
tokenized = raw.map(
    tokenize,
    num_proc=4,
    batched=True,
    # batch_size=500,
    remove_columns=["prompt", "completion"],
    desc="Tokenizing + extracting ratings/genres",
)
tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "rating_label", "genre_labels"],
)
tokenized["train"].save_to_disk("g32_head_train_tokenized_8k")
tokenized["validation"].save_to_disk("g32_head_validation_tokenized_8k")
tokenized["test"].save_to_disk("g32_head_test_tokenized_8k")
