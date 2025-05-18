# file for combining datasets

import json
import glob
import os

# Directory containing the JSON files
directory = os.path.dirname('results/')

# Find 10 JSON files (adjust the pattern if needed)
json_files = sorted(glob.glob(os.path.join(directory, '*lora_head_outputs.json')))[:10]

combined_data = []

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            combined_data.extend(data)
        else:
            combined_data.append(data)

with open('head_v2_combined.json', 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)