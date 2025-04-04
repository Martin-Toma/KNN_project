"""
Trims inner part of the subtitles

Autor: M. Tomasovic
"""
from openai import OpenAI
import json
from pathlib import Path
import random
import argparse

# argument parsing
parser = argparse.ArgumentParser(description="Trim content length in JSON dataset.")
parser.add_argument("--limit", type=int, required=True, help="Maximum allowed content length.")
args = parser.parse_args()

# prepare file reference
filePrompt = Path("split_dataset/test_subset_v2.json").resolve()

THRESH_LEN = args.limit  # load limit from argument

# load dataset in json format
with open(filePrompt, "r", encoding="utf-8") as pf:
    fileContents = json.load(pf)

    for i, content in enumerate(fileContents):
        if len(content['content']) > THRESH_LEN:
            print(f"Found too big content at {i}")

            original_len = len(content['content'])
            trimLen = original_len - THRESH_LEN

            start = 0
            # pick random start point it should be after in range 25% and 80%
            max_trim_idx = (original_len - trimLen) - (original_len // 5)
            if max_trim_idx <= original_len:
                start = random.randint(original_len // 4,  (original_len // 4) + trimLen)
            else:
                print("Too big trimm")
                exit(1)
            # trim out the section from the middle
            content['content'] = content['content'][:start] + content['content'][start + trimLen:]

            print("Trimmed")

            print(len(content['content']))
           
# save the modified dataset back to JSON file
output_file = Path(f"split_dataset/test_subset_v2_trimmed_{THRESH_LEN}.json").resolve()
with open(output_file, "w", encoding="utf-8") as pf:
    json.dump(fileContents, pf, ensure_ascii=False, indent=4)
