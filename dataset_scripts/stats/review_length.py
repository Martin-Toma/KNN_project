import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("../final_final_dataset.db")
rows = conn.execute("SELECT num,reviews FROM dataset").fetchall()
conn.close()
df = pd.DataFrame(rows, columns=["num", "reviews"])
df["reviews"] = df["reviews"].apply(json.loads)
review_lens = []


def add_review_len(x):
    for review in x["reviews"]:
        try:
            review_lens.append(len(review.split()))
        except:
            try:
                review_lens.append(len(review["text"].split()))
            except:
                pass


df["reviews"].apply(lambda x: add_review_len(x))
min_rl = min(review_lens)
max_rl = max(review_lens)
avg_rl = sum(review_lens) / len(review_lens)
limited_reviews_len = [x for x in review_lens if x < 450]
# print results
print(f"Min review word count: {min_rl}")
print(f"Max review word count: {max_rl}")
print(f"Average review word count: {avg_rl:.2f}")

# visualize the genre lengths distribution
plt.hist(limited_reviews_len, bins=100, color="blue", edgecolor="black", alpha=0.7)
plt.axvline(
    min_rl, color="red", linestyle="dashed", linewidth=2, label=f"Min: {min_rl}"
)
plt.axvline(
    max_rl, color="green", linestyle="dashed", linewidth=2, label=f"Max: {max_rl}"
)
plt.axvline(
    avg_rl, color="orange", linestyle="dashed", linewidth=2, label=f"Avg: {avg_rl:.2f}"
)
plt.xlim(0, 450)
plt.xlabel('Length of reviews(in "tokens")')
plt.ylabel("Frequency")
plt.title("Word count of reviews")
plt.legend()
plt.savefig("review_lengths.png")
plt.show()
# save the plot
