import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
conn = sqlite3.connect('../final_final_dataset.db')
rows = conn.execute('SELECT num, name,reviews FROM dataset').fetchall()
conn.close()
df = pd.DataFrame(rows, columns=["num","title","reviews"])
df['reviews'] = df['reviews'].apply(json.loads)
df["rating"] = df["reviews"].apply(lambda x: x["rating"])

#Print the top 10 most reviewed movies
top_10_movies = df.nlargest(10, 'rating')
print(top_10_movies[["title", "rating"]])
print("-------------------------------------------------------------")
# Print the movies with the least reviews
least_10_movies = df.nsmallest(10, 'rating')
print(least_10_movies[["title", "rating"]])