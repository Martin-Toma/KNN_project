from test_scores import test_rating_accuracy
from test_reviews import test_sentiment_alignment, test_review_similarity, test_grammar_accuracy

test_cases = load_test_dataset()  # Load test dataset: list of (subtitles, ground_truth_ratings, ground_truth_reviews)
predict_rating = []
ground_truth_rating = []
predict_review = []
ground_truth_review = []

for subtitles, true_rating, review in test_cases:
    pred_rating, pred_review = predict_rating(subtitles) # Predict rating using the LLM
    predict_rating.append(pred_rating)
    ground_truth_rating.append(true_rating)
    predict_review.append(pred_review)
    ground_truth_review.append(review)


mae, rmse, within_tolerance = test_rating_accuracy(predict_rating, ground_truth_rating)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, % within ±0.5: {within_tolerance*100:.2f}%")

avg_similarity = test_review_similarity(predict_review, ground_truth_review)
print(f"Avg. similarity to human reviews: {avg_similarity:.2f}")

mismatch_percentage = test_sentiment_alignment(predict_rating, predict_review)
print(f"Percentage of mismatched sentiment-rating pairs: {mismatch_percentage:.2f}%")

grammar_accuracy = test_grammar_accuracy(predict_review)
print(f"Average grammar errors in a review: {grammar_accuracy:.2f}")