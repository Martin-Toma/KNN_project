import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
import random

# Test function to check the similarity of generated reviews to original reviews
def test_review_similarity(movie_data):
    similarities = []
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for movie in movie_data:
        # Get all original review texts
        original_texts = [review for review in movie['original_reviews']]

        # Use the single generated review
        if movie.get("generated_review"):
            gen_text = movie["generated_review"]
            
            # Compare against all original reviews
            for orig_text in original_texts:
                if len(orig_text) < 50 or orig_text is None or gen_text is None:  # Skip short reviews
                    continue
                orig_embed = sbert_model.encode(orig_text)
                gen_embed = sbert_model.encode(gen_text)
                similarity = cosine_similarity([orig_embed], [gen_embed])[0][0]
                similarities.append(similarity)
    
    return np.mean(similarities) if similarities else 0, similarities
    
def test_orig_review_similarity(movie_data):
    similarities = []
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    for movie in movie_data:
        # Filter out short or None reviews
        original_texts = [review for review in movie['original_reviews'] if review and len(review) >= 50]

        # Compare each review to the next one
        for i in range(len(original_texts) - 1):
            orig_text = original_texts[i]
            next_text = original_texts[i + 1]

            # Encode both reviews
            orig_embed = sbert_model.encode(orig_text)
            next_embed = sbert_model.encode(next_text)

            # Calculate cosine similarity
            similarity = cosine_similarity([orig_embed], [next_embed])[0][0]
            similarities.append(similarity)

    return np.mean(similarities) if similarities else 0, similarities

def test_cross_movie_review_similarity(movie_data):
    similarities = []
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    for i in range(len(movie_data) - 1):
        current_movie = movie_data[i]
        next_movie = movie_data[i + 1]

        # Filter out short or None reviews
        current_reviews = [r for r in current_movie['original_reviews'] if r and len(r) >= 50]
        next_reviews = [r for r in next_movie['original_reviews'] if r and len(r) >= 50]

        if not current_reviews or not next_reviews:
            continue

        # Pick a random review from the current movie
        chosen_review = random.choice(current_reviews)
        chosen_embed = sbert_model.encode(chosen_review)

        # Compare to all reviews from the next movie
        for next_review in next_reviews:
            next_embed = sbert_model.encode(next_review)
            similarity = cosine_similarity([chosen_embed], [next_embed])[0][0]
            similarities.append(similarity)

    return np.mean(similarities) if similarities else 0, similarities

# Test function to check the sentiment alignment of generated reviews with their generated ratings
def test_sentiment_alignment(movie_data):
    analyzer = SentimentIntensityAnalyzer()
    mismatches = 0
    alignment_diff = []
    total = 0
    
    for movie in movie_data:
        if "generated_review" in movie and "generated_rating" in movie:
            rating = movie["generated_rating"]
            text = movie["generated_review"]

            if rating is None or text is None:
                continue

            # Calculate sentiment using VADER
            vader_sentiment = analyzer.polarity_scores(text)['compound']
            expected_sentiment = (rating - 5) / 5  # Scale 0-10 to [-1, 1]
            diff = vader_sentiment - expected_sentiment
            alignment_diff.append(diff)
            
            if abs(diff) >= 0.5:
                mismatches += 1
            total += 1
                
    return (mismatches / total) * 100 if total > 0 else 0, alignment_diff

# Test function to check the grammar accuracy of generated reviews
def test_grammar_accuracy(movie_data):
    tool = language_tool_python.LanguageTool('en-US')
    total_errors = 0
    errors_num = []
    total_reviews = 0
    
    for movie in movie_data:
        if "generated_review" in movie and movie["generated_review"]:
            try:
                matches = tool.check(movie["generated_review"])
                total_errors += len(matches)
                errors_num.append(len(matches))
                total_reviews += 1
            except Exception as e:
                print(f"Error checking review for movie {movie['num']}: {e}")
    
    return total_errors / total_reviews if total_reviews > 0 else 0, errors_num
