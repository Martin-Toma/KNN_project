from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
import language_tool_python

def test_review_similarity(gen_reviews, human_reviews):
    similarities = []
    # Initialize Sentence-BERT for similarity
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    for generated_review, human_review in zip(gen_reviews, human_reviews):        
        human_embedding = sbert_model.encode(human_review)
        generated_embedding = sbert_model.encode(generated_review)
        similarity = cosine_similarity([human_embedding], [generated_embedding])[0][0]
        similarities.append(similarity)

    avg_similarity = np.mean(similarities)
    return avg_similarity

def test_sentiment_alignment(ratings, reviews):
    # Use VADER or TextBlob to check if review sentiment matches rating
    mismatches = 0

    for rating, review in zip(ratings, reviews):
        sentiment = TextBlob(review).sentiment.polarity
        expected_sentiment = (rating - 2.5) / 2.5  # Scale 0-5 to [-1, 1]
        if abs(sentiment - expected_sentiment) >= 0.3:
            mismatches += 1

    total_reviews = len(reviews)
    mismatch_percentage = (mismatches / total_reviews) * 100
    return mismatch_percentage

def test_grammar_accuracy(reviews):
    # Use LanguageTool to check grammar accuracy
    tool = language_tool_python.LanguageTool('en-US')
    errors_num = 0

    for review in reviews:
        matches = tool.check(review)
        errors_num += len(matches)
    
    return errors_num / len(reviews)