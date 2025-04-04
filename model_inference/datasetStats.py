"""
For the dataset stored in SQL database, plots statistics

Autor: M. Tomasovic
"""
import sqlite3
import numpy as np
import json
import matplotlib.pyplot as plt

# Connect to the SQLite database
dbConnect = sqlite3.connect('final_final_dataset.db')
dbConnect.row_factory = sqlite3.Row  # Allow fetching rows as dictionaries
curs = dbConnect.cursor()

# Extract a random subset of rows including the reviews column
curs.execute('''
SELECT
  d.num,
  d.name,
  d.content,
  d.reviews,
  d.genres
FROM dataset d
''')

# fetch all rows
rows = curs.fetchall()

# list to store extracted ratings
ratings = []

# iterate over the rows to extract ratings
for row in rows:
    reviews_json = row['reviews']  
    
    try:
        # parse the JSON string
        review_data = json.loads(reviews_json)
        
        # extract the rating
        rating = review_data.get('rating')
        
        # if the rating is a valid float add it to the list
        if isinstance(rating, (float, int)):
            ratings.append(rating)
    
    except json.JSONDecodeError:
        # handle cases where the 'reviews' column is not valid JSON
        print(f"Invalid JSON in row {row['num']}")

# calculate statistics on the ratings
if ratings:
    avg_rating = np.mean(ratings)
    min_rating = np.min(ratings)
    max_rating = np.max(ratings)
    median_rating = np.median(ratings)
    rating_count = len(ratings)
    rating_stddev = np.std(ratings)
    
    # print the statistics
    print(f"Total ratings found: {rating_count}")
    print(f"Average rating: {avg_rating:.2f}")
    print(f"Median rating: {median_rating:.2f}")
    print(f"Standard Deviation: {rating_stddev:.2f}")
    print(f"Min rating: {min_rating}")
    print(f"Max rating: {max_rating}")

    # visualize ratings distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ratings, bins=20, color='blue', edgecolor='black', alpha=0.7)
    
    # add dashed lines for min, max, and average ratings
    plt.axvline(min_rating, color='red', linestyle='dashed', linewidth=2, label=f"Min: {min_rating}")
    plt.axvline(max_rating, color='green', linestyle='dashed', linewidth=2, label=f"Max: {max_rating}")
    plt.axvline(avg_rating, color='orange', linestyle='dashed', linewidth=2, label=f"Avg: {avg_rating:.2f}")
    
    # add labels and title
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Average Ratings")
    
    # add legend
    plt.legend()
    
    # show the plot
    plt.show()
else:
    print("No valid ratings found.")

# close the database connection
dbConnect.close()