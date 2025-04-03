from openai import OpenAI
import json
from pathlib import Path
import random

# prepare file reference
filePrompt = Path("split_dataset/test_subset_v2.json").resolve()

THRESH_LEN = 129000

# load dataset in json format
with open(filePrompt, "r", encoding="utf-8") as pf:
    fileContents = json.load(pf)

    for i, content in enumerate(fileContents):
        if len(content['content']) > THRESH_LEN:
            print(f"Found too big content at {i}")

            original_len = len(content['content'])
            trimLen = original_len - THRESH_LEN

            # pick random start point within the first 40% to 60% range
            start = random.randint(original_len // 4, (3 * original_len // 4) - trimLen)

            # trim out the section from the middle
            content['content'] = content['content'][:start] + content['content'][start + trimLen:]

            print("Trimmed")

            print(len(content['content']))
           
# save the modified dataset back to JSON file
output_file = Path("split_dataset/test_subset_v2_trimmed.json").resolve()
with open(output_file, "w", encoding="utf-8") as pf:
    json.dump(fileContents, pf, ensure_ascii=False, indent=4)
