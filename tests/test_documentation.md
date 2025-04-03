# Testing

## Score tests
For evaluating the models ability to predict the scores of movies three basic metrics were used MAE, RMSE, and % within tolerance.

    Mean Absolute Error - MAE: Measures the average absolute difference between predicted and actual scores.

    Root Mean Squared Error - RMSE: Measures the squared absolute difference, meaning it punishes larger errors more heavily.

    % Within Tolerance: Measures the proportion of predictions falling within ±1 points of the true score.

## Review tests
To ensure the quality and coherence of the generated movie reviews, the following tests were implemented:

### Grammar test
Checks the grammatical accuracy of generated reviews using LanguageTool - an open-source grammar and style which detects grammatical errors, punctuation mistakes, and stylistic issues.

The metric for this test is the average number of errors per review.

### Review similarity test
Measures how semantically similar generated reviews are to real human-written reviews using SBERT (Sentence-BERT).

We use the all-MiniLM-L6-v2 model from SentenceTransformers to convert both generated and original reviews into high-dimensional vectors. After encoding the texts, we compute the cosine similarity between the generated review and each original human-written review. Higher similarity scores indicate that the generated review conveys similar ideas and meanings to the original reviews.

The metric for this  is the average similarity to a human review from 0 to 1.

Human written reviews with fewer than 15 characters are automatically skipped over and are not tested.

### Sentiment alighment test
Verifies whether the sentiment of a generated review matches its given rating.

To analyze the review text and assigns a sentiment score we use VADER - Valence Aware Dictionary and sEntiment Reasoner, A rule-based sentiment analysis tool which utputs a compound score between -1 (negative) and +1 (positive).

Movie scores are scaled to [-1, 1] and copared to VADER's score. If the difference between expected and predicted sentiment exceeds 0.5, it’s flagged as a mismatch.

The metric for this test is the percentage of reviews with mismatched sentiment.

## Genres tests

Evaluates the model’s ability to correctly predict movie genres by comparing the generated genres to the original human-assigned genres.

We use standard classification metrics which are:

    Precision: Measures the proportion of correctly predicted genres out of all predicted genres.

    Recall: Measures the proportion of correctly predicted genres out of all actual genres.

    F1-score: The harmonic mean of precision and recall, balancing the two metrics.

These scores help assess how well the model generates relevant genres without adding incorrect ones. 