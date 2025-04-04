"""
Perplexity calculation for huggingface models

Autor: M. Tomasovic
"""
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = "tiiuae/Falcon3-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cpu'#'cuda'
model.to(device)

# load dataset in json format
dataset_pth = 'test_subset_v2_trimmed_63000.json'
with open(dataset_pth, "r", encoding="utf-8") as pf:
    dataset = json.load(pf)

# extract subtitle part
subtitles = ['Hello I am Martin', "Hi how are you"] #[item['content'] for item in dataset if 'content' in item]

encodings = tokenizer(subtitles, return_tensors="pt", padding=True, truncation=True,)

print(encodings)

max_length = 2048 #model.max_seq_len
stride = 512
seq_len = encodings.input_ids.size(1)

print(encodings)


# inspired by: https://huggingface.co/docs/transformers/perplexity
nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    # Accumulate the total negative log-likelihood and the total number of tokens
    num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
    batch_size = target_ids.size(0)
    num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
    nll_sum += neg_log_likelihood * num_loss_tokens
    n_tokens += num_loss_tokens

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
ppl = torch.exp(avg_nll)

print(ppl)