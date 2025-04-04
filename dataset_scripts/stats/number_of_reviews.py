import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
conn = sqlite3.connect('../final_final_dataset.db')
rows = conn.execute('SELECT num,reviews,name FROM dataset').fetchall()
conn.close()
df = pd.DataFrame(rows, columns=["num","reviews","name"])
df['reviews'] = df['reviews'].apply(json.loads)
df["reviews_count"] = df["reviews"].apply(lambda x: len(x["reviews"]))

min_rl = min(df["reviews_count"])
max_rl = max(df["reviews_count"])
df["limited_reviews_count"] = df["reviews_count"].apply(lambda x: x if x < 150 else None)
df["limited_reviews_count"].dropna(inplace=True)
avg_rl = df["reviews_count"].mean()

# print results
print(f"Min reviews: {min_rl}")
print(f"Max reviews: {max_rl}")
print(f"Average reviews: {avg_rl:.2f}")
print(f"Total reviews: {df['reviews_count'].sum()}")
# visualize the genre lengths distribution
plt.xlim(0, 150)
plt.hist(df["limited_reviews_count"], bins=200, color='blue', edgecolor='black', alpha=0.7)
plt.axvline(min_rl, color='red', linestyle='dashed', linewidth=2, label=f"Min: {min_rl}")
plt.axvline(max_rl, color='green', linestyle='dashed', linewidth=2, label=f"Max: {max_rl}")
plt.axvline(avg_rl, color='orange', linestyle='dashed', linewidth=2, label=f"Avg: {avg_rl:.2f}")
plt.xlabel("Number of reviews per movie")
plt.ylabel("Frequency")
plt.title("Distribution of reviews per movie")
plt.legend()
plt.savefig("number_of_reviews.png")
plt.show()
# save the plot
# Get the 10 rows with the lowest and highest reviews count
min_rows = df.nsmallest(10, 'reviews_count')
max_rows = df.nlargest(10, 'reviews_count')
# Save to JSON files
min_rows.to_json("min_reviews.json", orient="records", lines=True)
max_rows.to_json("max_reviews.json", orient="records", lines=True)