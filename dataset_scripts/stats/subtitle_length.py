import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
conn = sqlite3.connect('../final_final_dataset.db')
rows = conn.execute('SELECT num,content,name FROM dataset').fetchall()
conn.close()
df = pd.DataFrame(rows, columns=["num","subtitle","name"])

df['subtitle_word_count'] = df['subtitle'].apply(lambda x: len(x.split()))
data = df['subtitle_word_count'].dropna()
# Drop the ones shorter than 15 words
data = data[data >= 15]
df = df[df['subtitle_word_count'] >= 15]
min_rl = min(df["subtitle_word_count"])
max_rl = max(df["subtitle_word_count"])
avg_rl = df["subtitle_word_count"].mean()
kde = gaussian_kde(data)
x_vals = np.linspace(data.min(), data.max(), 1000)
kde_vals = kde(x_vals)
# print results
print(f"Min subtitle word count: {min_rl}")
print(f"Max subtitle word count: {max_rl}")
print(f"Average subtitle word count: {avg_rl:.2f}")

# visualize the genre lengths distribution
# counts, bin_edges = np.histogram(df['subtitle_word_count'], bins=100)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# plt.plot(bin_centers, counts, color='blue', linewidth=1.5)
plt.xlim(0, 40000)
plt.plot(x_vals, kde_vals, color='black', linewidth=2, label="KDE")
plt.hist(df['subtitle_word_count'], bins=100, color='blue', edgecolor='black', alpha=0.6, density=True)
plt.axvline(min_rl, color='red', linestyle='dashed', linewidth=2, label=f"Min: {min_rl}")
plt.axvline(max_rl, color='green', linestyle='dashed', linewidth=2, label=f"Max: {max_rl}")
plt.axvline(avg_rl, color='orange', linestyle='dashed', linewidth=2, label=f"Avg: {avg_rl:.2f}")
plt.xlabel("Length of a subtitle(in \"tokens\")")
plt.ylabel("Density")
plt.title("Word count of subtitles")
plt.legend()
plt.savefig("subtitle_length.png")
plt.show()
# save the plot
# Get the 10 rows wit the lowest and highest subtitle word count
min_rows = df.nsmallest(10, 'subtitle_word_count')
max_rows = df.nlargest(10, 'subtitle_word_count')
# Save to JSON files
min_rows.to_json("min_subtitle.json", orient="records", lines=True)
max_rows.to_json("max_subtitle.json", orient="records", lines=True)