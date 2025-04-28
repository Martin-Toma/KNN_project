import json
from transformers import AutoTokenizer


MAX_IN_TOKENS = 31000 # 31k leaving some space free for just in case

full_datasets = ["2instruction_eval_dataset.json", "2instruction_test_dataset.json", "2instruction_train_dataset.json"]

def get_answer_size(data):
    max = 0
    maxResp = ""
    for d in data:
        n = d["completion"]
        lenN = len(n.split())
        if lenN > max:
            max = lenN
            maxResp = n
    print(max)
    print(maxResp)

from huggingface_hub import login

ac = ""

with open("hf_token.txt", 'r') as af:
    ac = af.read()

login(token = ac)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt") # load tokenizer

# actually cuts the middle part of the dataset leaving start and end intact
def cut_middle(strData, maxSize):
    firstHalf = maxSize // 2
    secondHalf = maxSize - firstHalf

    tokens = tokenizer.encode(strData, add_special_tokens=False)

    if len(tokens) <= maxSize:
        return strData  # no cut
    # otherwise cut
    tokensTrimmed = tokens[:firstHalf] + tokens[-secondHalf:]
    # decode back to text
    decoded = tokenizer.decode(tokensTrimmed, skip_special_tokens=True)
    return decoded

# make the prompt to be maximally maxSize tokens long to fit context size
def reduce_size(data, maxSize):
    for i in range(len(data)):
        data[i]["prompt"] = cut_middle(data[i]["prompt"], maxSize)
    return data

for dset in full_datasets:
    print("Reducing " + dset)
    # load dataset in json format
    with open(dset, "r", encoding="utf-8") as pf:
        fileContents = json.load(pf)

        #get_answer_size(fileContents)

        trimmed = reduce_size(fileContents, MAX_IN_TOKENS)

        with open(("small" + dset), "w", encoding="utf-8") as wf:
            json.dump(trimmed, wf, ensure_ascii=False, indent=4)