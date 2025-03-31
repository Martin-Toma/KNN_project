import numpy as np

def test_rating_accuracy(movie_data):
    all_preds = []
    all_truth = []
    
    for movie in movie_data:
        # Get all original ratings for this movie
        truth_ratings = [review['rating'] for review in movie['original_reviews']]
        
        # Compare each generated rating to all original ratings
        for gen_review in movie['generated_reviews']:
            all_preds.extend([gen_review['rating']] * len(truth_ratings))
            all_truth.extend(truth_ratings)
    
    # Calculate metrics using all comparisons
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_truth)))
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_truth)) ** 2))
    within_tolerance = np.mean(np.abs(np.array(all_preds) - np.array(all_truth)) <= 0.5)
    
    return mae, rmse, within_tolerance