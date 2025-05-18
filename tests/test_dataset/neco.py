import json

def remove_content_field(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    if isinstance(data, dict):
        data.pop("content", None)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                item.pop("content", None)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

# Example usage:
remove_content_field('test_subset_v2_trimmed.json', 'test_subset.json')