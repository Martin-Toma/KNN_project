from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import language_tool_python
import numpy as np

def test_review_similarity(movie_data):
    similarities = []
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for movie in movie_data:
        # Get all original review texts
        original_texts = [review['text'] for review in movie['original_reviews']]
        
        # Get generated review
        if movie['generated_reviews']:
            gen_review = movie['generated_reviews'][0]
            gen_text = gen_review['text']
            
            # Compare against all original reviews
            for orig_text in original_texts:
                orig_embed = sbert_model.encode(orig_text)
                gen_embed = sbert_model.encode(gen_text)
                similarity = cosine_similarity([orig_embed], [gen_embed])[0][0]
                similarities.append(similarity)
    
    return np.mean(similarities) if similarities else 0

def test_sentiment_alignment(movie_data):
    analyzer = SentimentIntensityAnalyzer()
    mismatches = 0
    total = 0
    
    for movie in movie_data:
        if movie['generated_reviews']:
            gen_review = movie['generated_reviews'][0]
            rating = gen_review['rating']
            text = gen_review['text']
            
            # Calculate sentiment using VADER
            vader_sentiment = analyzer.polarity_scores(text)['compound']
            expected_sentiment = (rating - 2.5) / 2.5  # Scale 0-5 to [-1, 1]
            
            if abs(vader_sentiment - expected_sentiment) >= 0.4:
                mismatches += 1
            total += 1
                
    return (mismatches / total) * 100 if total > 0 else 0

def test_grammar_accuracy(movie_data):
    tool = language_tool_python.LanguageTool('en-US')
    total_errors = 0
    total_reviews = 0
    
    for movie in movie_data:
        if movie['generated_reviews']:
            gen_review = movie['generated_reviews'][0]
            matches = tool.check(gen_review['text'])
            total_errors += len(matches)
            total_reviews += 1
    
    return total_errors / total_reviews if total_reviews > 0 else 0