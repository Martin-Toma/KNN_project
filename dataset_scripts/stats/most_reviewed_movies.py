import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
conn = sqlite3.connect('../final_final_dataset.db')
rows = conn.execute('SELECT num, name,reviews FROM dataset').fetchall()
conn.close()
df = pd.DataFrame(rows, columns=["num","title","reviews"])
df['reviews'] = df['reviews'].apply(json.loads)
df["reviews_count"] = df["reviews"].apply(lambda x: len(x["reviews"]))

#Print the top 10 most reviewed movies
top_10_movies = df.nlargest(10, 'reviews_count')
print(top_10_movies[["title", "reviews_count"]])
print("-------------------------------------------------------------")
# Print the movies with the least reviews
least_10_movies = df.nsmallest(10, 'reviews_count')
print(least_10_movies[["title", "reviews_count"]])