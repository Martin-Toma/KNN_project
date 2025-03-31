from test_scores import test_rating_accuracy
from test_reviews import test_sentiment_alignment, test_review_similarity, test_grammar_accuracy
from get_data import load_test_dataset

file1 = 'test_dataset/test_subset.json'
file2 = 'test_dataset/groupedOutJustJSON.json'

test_cases = load_test_dataset(file1, file2)

if test_cases is None:
    print("Failed to load test cases.")
    exit(1)

mae, rmse, within_tolerance = test_rating_accuracy(test_cases)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, % within ±0.5: {within_tolerance*100:.2f}%")

avg_similarity = test_review_similarity(test_cases)
print(f"Avg. similarity to human reviews: {avg_similarity:.2f}")

mismatch_percentage = test_sentiment_alignment(test_cases)
print(f"Percentage of mismatched sentiment-rating pairs: {mismatch_percentage:.2f}%")

grammar_accuracy = test_grammar_accuracy(test_cases)
print(f"Average grammar errors in a review: {grammar_accuracy:.2f}")

with open('evaluation_results.txt', 'w') as f:
    f.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, % within ±0.5: {within_tolerance*100:.2f}%\n")
    f.write(f"Avg. similarity to human reviews: {avg_similarity:.2f}\n")
    f.write(f"Percentage of mismatched sentiment-rating pairs: {mismatch_percentage:.2f}%\n")
    f.write(f"Average grammar errors in a review: {grammar_accuracy:.2f}\n")