import numpy as np

def test_rating_accuracy(movie_data):
    all_preds = []
    all_truth = []
    
    for movie in movie_data:
        # Ensure both ratings exist
        if movie["generated_rating"] is not None and movie["original_rating"] is not None:
            all_preds.append(movie["generated_rating"])
            all_truth.append(movie["original_rating"])
    
    if not all_preds or not all_truth:
        return None, None, None, None  # Handle case where no valid ratings exist
    
    # Convert to NumPy arrays for calculations
    all_preds = np.array(all_preds)
    all_truth = np.array(all_truth)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_preds - all_truth))
    rmse = np.sqrt(np.mean((all_preds - all_truth) ** 2))
    within_tolerance = np.mean(np.abs(all_preds - all_truth) <= 1)
    all_diffs = all_preds - all_truth
    
    return mae, rmse, within_tolerance, all_diffs
