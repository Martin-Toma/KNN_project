from test_scores import test_rating_accuracy
from test_reviews import test_sentiment_alignment, test_review_similarity, test_grammar_accuracy
from get_data import load_test_dataset
from test_genres import test_genre_accuracy
import matplotlib.pyplot as plt
import numpy as np

file1 = 'test_dataset/test_subset_v2_trimmed.json'
file2 = 'test_dataset/groupedOutJustJSONv2.json'

test_cases = load_test_dataset(file1, file2)

if test_cases is None:
    print("Failed to load test cases.")
    exit(1)

mae, rmse, within_tolerance, score_diff = test_rating_accuracy(test_cases)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, % within ±1: {within_tolerance*100:.2f}%")
# Create a histogram for score_diff
plt.figure(figsize=(10, 6))
plt.hist(score_diff, bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribution of Score Differences')
plt.xlabel('Score Difference')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('test_results/score_diff_distribution.png')

avg_similarity, similarities = test_review_similarity(test_cases)
print(f"Avg. similarity to human reviews: {avg_similarity:.2f}")
# Create a histogram for similarities
plt.figure(figsize=(10, 6))
plt.hist(similarities, bins=20, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of Review Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.savefig('test_results/review_similarity_distribution.png')

# TODO maybe remove
mismatch_percentage, mismatches = test_sentiment_alignment(test_cases)
print(f"Percentage of mismatched sentiment-rating pairs: {mismatch_percentage:.2f}%")
# Create a histogram for sentiment alignment differences
plt.figure(figsize=(10, 6))
plt.hist(mismatches, bins=10, color='red', alpha=0.7, edgecolor='black')
plt.title('Distribution of Sentiment Alignment Differences')
plt.xlabel('Sentiment Alignment Difference')
plt.ylabel('Frequency')
plt.savefig('test_results/sentiment_alignment_distribution.png')

grammar_accuracy, errors_num = test_grammar_accuracy(test_cases)
print(f"Average grammar errors in a review: {grammar_accuracy:.2f}")
# Create a histogram for grammar errors
plt.figure(figsize=(10, 6))
# Ensure that the bins are set to whole numbers for the histogram
unique_errors = np.unique(errors_num)
plt.hist(errors_num, bins=np.arange(min(unique_errors), max(unique_errors) + 1), color='purple', alpha=0.7, edgecolor='black')
plt.title('Distribution of Grammar Errors in Reviews')
plt.xlabel('Number of Grammar Errors')
plt.ylabel('Frequency')
plt.savefig('test_results/grammar_errors_distribution.png')

avg_precision, avg_recall, f1_score, precisions, recalls, f1_scores = test_genre_accuracy(test_cases)
print(f"Genre Accuracy - Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1 Score: {f1_score:.2f}")
# Create Histogram
plt.figure(figsize=(10, 6))
bins = [i / 10 for i in range(11)]  # Bins from 0.0 to 1.0 in steps of 0.1
plt.hist(precisions, bins=bins, alpha=0.6, color='blue', label="Precision", histtype="stepfilled")
plt.hist(recalls, bins=bins, alpha=0.6, color='green', label="Recall", histtype="stepfilled")
plt.hist(f1_scores, bins=bins, alpha=0.6, color='red', label="F1 Score", histtype="stepfilled")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Precision, Recall, and F1 Scores")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig('test_results/precision_recall_f1_distribution.png')

with open('test_results/evaluation_results.txt', 'w') as f:
    f.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, % within ±1: {within_tolerance*100:.2f}%\n")
    f.write(f"Avg. similarity to human reviews: {avg_similarity:.2f}\n")
    f.write(f"Percentage of mismatched sentiment-rating pairs: {mismatch_percentage:.2f}%\n")
    f.write(f"Average grammar errors in a review: {grammar_accuracy:.2f}\n")
    f.write(f"Genre Accuracy - Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1 Score: {f1_score:.2f}\n")