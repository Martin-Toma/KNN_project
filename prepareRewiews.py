"""
Prepares review part of dataset. Format is title, average_rating, cleaned_text

Autor: M. Tomasovic
"""
from datasets import load_dataset
import json

# load dataset
dataset = load_dataset("rohan2810/amazon-movies-meta-reviews-merged")

# extract relevant columns - title, average_rating, cleaned_text
selected_data = [
    {
        'title': row['title'],
        'average_rating': row['average_rating'],
        'cleaned_text': row['cleaned_text']
    }
    for row in dataset['train']
]

# save data in json format
output_file_path = 'amazon_movies_data.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(selected_data, f, ensure_ascii=False, indent=4)

print("Data saved to: ", output_file_path)
