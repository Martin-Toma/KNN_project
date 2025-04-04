"""
This script plots and prints statistics review amd rating.

Autor: M. Tomasovic
"""
import matplotlib.pyplot as plt
from datasets import load_dataset

# load dataset with progress bar
dataset = load_dataset("rohan2810/amazon-movies-meta-reviews-merged")

# extract average ratings with progress visualization
ratings = dataset['train']['average_rating']

# extract and clean average ratings
ratings = [r for r in dataset['train']['average_rating'] if r is not None]

if ratings:  # Ensure list is not empty
    min_rating = min(ratings)
    max_rating = max(ratings)
    avg_rating = sum(ratings) / len(ratings)

    # print results
    print(f"Min rating: {min_rating}")
    print(f"Max rating: {max_rating}")
    print(f"Average rating: {avg_rating:.2f}")

    # visualize ratings distribution
    plt.hist(ratings, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(min_rating, color='red', linestyle='dashed', linewidth=2, label=f"Min: {min_rating}")
    plt.axvline(max_rating, color='green', linestyle='dashed', linewidth=2, label=f"Max: {max_rating}")
    plt.axvline(avg_rating, color='orange', linestyle='dashed', linewidth=2, label=f"Avg: {avg_rating:.2f}")
    plt.xlabel("Average Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Average Ratings")
    plt.legend()
    plt.show()
else:
    print("No valid ratings found.")

review_len = [r for r in dataset['train']['review_length'] if r is not None and r < 1000]

if review_len:  # ensure list is not empty
    min_rl = min(review_len)
    max_rl = max(review_len)
    avg_rl = sum(review_len) / len(review_len)

    # print results
    print(f"Min review length: {min_rl}")
    print(f"Max review length: {max_rl}")
    print(f"Average review length: {avg_rl:.2f}")

    # visualize ratings distribution
    plt.hist(review_len, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(min_rl, color='red', linestyle='dashed', linewidth=2, label=f"Min: {min_rl}")
    plt.axvline(max_rl, color='green', linestyle='dashed', linewidth=2, label=f"Max: {max_rl}")
    plt.axvline(avg_rl, color='orange', linestyle='dashed', linewidth=2, label=f"Avg: {avg_rl:.2f}")
    plt.xlabel("Average Review length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Review lengths")
    plt.legend()
    plt.show()