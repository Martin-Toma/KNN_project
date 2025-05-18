import json
import sys

def calc_avg_perplexity(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    perplexities = [item['perplexity'] for item in data if 'perplexity' in item]
    avg = sum(perplexities) / len(perplexities) if perplexities else 0
    print(f"Average perplexity: {avg}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calc_avg.py <json_file>")
    else:
        calc_avg_perplexity(sys.argv[1])