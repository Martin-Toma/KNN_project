import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

conn = sqlite3.connect("../final_final_dataset.db")
rows = conn.execute("SELECT num,genres FROM dataset").fetchall()
conn.close()
df = pd.DataFrame(rows, columns=["num", "genres"])
# Drop NA
df = df.dropna()
df["split_genres"] = df["genres"].apply(lambda x: x.split(","))

# Create a co-occurrence matrix
# Flatten the list to get unique genres
all_genres = sorted(set(g for sublist in df["split_genres"] for g in sublist))
genre_index = {genre: idx for idx, genre in enumerate(all_genres)}

# Initialize the co-occurrence matrix
matrix = np.zeros((len(all_genres), len(all_genres)), dtype=int)

# Fill the co-occurrence matrix
for genres in df["split_genres"]:
    if len(genres) == 1:
        # idx = genre_index[genres[0]]
        # matrix[idx][idx] += 1
        pass
    else:
        for i in range(len(genres)):
            for j in range(i, len(genres)):
                if i == j:
                    continue
                idx_i = genre_index[genres[i]]
                idx_j = genre_index[genres[j]]
                matrix[idx_i][idx_j] += 1
                if i != j:
                    matrix[idx_j][idx_i] += 1
# Convert to DataFrame for easier viewing
co_occurrence_df = pd.DataFrame(matrix, index=all_genres, columns=all_genres)

# Display top of the matrix
print(co_occurrence_df.head())

# Optionally: visualize as heatmap
plt.figure(figsize=(10, 8))
plt.imshow(co_occurrence_df, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Co-occurrence Count")
plt.xticks(ticks=np.arange(len(all_genres)), labels=all_genres, rotation=90)
plt.yticks(ticks=np.arange(len(all_genres)), labels=all_genres)
plt.title("Genre Co-occurrence Matrix")
plt.tight_layout()
plt.savefig("genre_co_occurrence_matrix.png")
plt.show()
