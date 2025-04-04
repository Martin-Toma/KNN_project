"""
Prepares review part of dataset. Format is title, average_rating, cleaned_text

Autor: M. Tomasovic, A. Kovacs
"""

from datasets import load_dataset

# SLOW
# def get_reviews():
#     # load dataset
#     dataset = load_dataset("rohan2810/amazon-movies-meta-reviews-merged")
#     selected_data = {}
#     for row in dataset["train"]:
#         if not selected_data.get(row["title"]):
#             selected_data[row["title"]] = []
#         selected_data[row["title"]].append(row["cleaned_text"])
#     return selected_data


def get_reviews():
    dataset = load_dataset("rohan2810/amazon-movies-meta-reviews-merged")

    # Convert to pandas for faster groupby operation
    df = dataset["train"].to_pandas()
    # Group by title and aggregate reviews
    grouped = df.groupby("title")["cleaned_text"].apply(list).to_dict()
    return grouped
