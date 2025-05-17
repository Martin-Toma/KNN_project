"""
reduce_samples.py
=================
Description: reduces from 5 to 3 samples per movie (3 different reviews)

Author: Martin Tomasovic
"""

import json

splits = ['test_32'] #, 'validation_32', 'train_32'] # all dataset splits

from collections import defaultdict

def reduce_reviews(data):
    grouped = defaultdict(list)
    
    # group reviews by prompt
    for item in data:
        grouped[item["prompt"]].append(item)
    
    # keep only first 3 per group
    result = []
    for reviews in grouped.values():
        result.extend(reviews[:1])
    
    return result

for split in splits:

    # load dataset in json format
    splitJsonPth = split  + ".json"                 # load path
    splitNewJsonPth = "test1_" + splitJsonPth # store path
    with open(splitJsonPth, "r", encoding="utf-8") as pf:
        allData = json.load(pf)

        # create prompts for each movie max 3 with different reviews
        dset = reduce_reviews(allData)

        # store the result
        with open(splitNewJsonPth, "w", encoding="utf-8") as wf:
            json.dump(dset, wf, ensure_ascii=False, indent=4)
        