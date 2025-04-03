import sqlite3
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pth = r"C:\Users\marti\Music\knn\KNN_project\final_final_dataset.db"

dbConnect = sqlite3.connect(pth)
dbConnect.row_factory = sqlite3.Row 
curs = dbConnect.cursor()

curs.execute('''
SELECT * FROM dataset d
''')

# extract genres, split by commas to create lists of genres
genres = [str(row["genres"]).split(',') for row in curs.fetchall()]

# flatten the list of genres so that we have all genres in a single list
flat_genres = [genre for sublist in genres for genre in sublist]

unique_genres = np.unique(flat_genres)

print(unique_genres)

""" length stats """
genres = [row["genres"] for row in curs.fetchall()]  # extract genres as a list of strings

# genres are a comma-separated string
genre_lengths = [len(str(genre).split(',')) for genre in genres] 

print(genre_lengths)

# calculate stats
if genre_lengths:  # ensure list is not empty
    min_rl = min(genre_lengths)
    max_rl = max(genre_lengths)
    avg_rl = sum(genre_lengths) / len(genre_lengths)

    # print results
    print(f"Min review length: {min_rl}")
    print(f"Max review length: {max_rl}")
    print(f"Average review length: {avg_rl:.2f}")

    # visualize the genre lengths distribution
    plt.hist(genre_lengths, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(min_rl, color='red', linestyle='dashed', linewidth=2, label=f"Min: {min_rl}")
    plt.axvline(max_rl, color='green', linestyle='dashed', linewidth=2, label=f"Max: {max_rl}")
    plt.axvline(avg_rl, color='orange', linestyle='dashed', linewidth=2, label=f"Avg: {avg_rl:.2f}")
    plt.xlabel("Number of Genres")
    plt.ylabel("Frequency")
    plt.title("Distribution of Genre Counts")
    plt.legend()
    plt.show()
