import numpy as np

# Test function to evaluate the accuracy of genre predictions
def test_genre_accuracy(movie_data):
    precisions = []
    recalls = []
    f1_scores = []
    
    for movie in movie_data:
        original_genres = set(movie["original_genres"])
        generated_genres = set(movie["generated_genres"])
        
        if not original_genres or not generated_genres:
            continue  # Skip movies with missing genre info
        
        intersection = original_genres.intersection(generated_genres)
        
        precision = len(intersection) / len(generated_genres) if generated_genres else 0
        recall = len(intersection) / len(original_genres) if original_genres else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    f1_score = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, f1_score, precisions, recalls, f1_scores
