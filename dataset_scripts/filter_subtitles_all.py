import pandas as pd

# Read the TSV file and skip lines that don't have the expected number of columns
df = pd.read_csv("subtitles_all.txt", sep="\t", dtype=str, on_bad_lines="skip")

# Filter rows: keep only rows where ISO639 (lowercased) is either "en" or "eng"
filtered_df = df[df["ISO639"].str.lower().isin(["en", "eng"])]
# Remove serials uncategorized movies (the reviews dataset doesn't differentiate between episodes of TV shows)
only_movies = filtered_df.query("MovieKind == 'movie'")
# Remove duplicates based on MovieName (Not used because some .zip files don't contain subtitles)
# only_movies = only_movies.drop_duplicates(subset=["MovieName"])
# Write the filtered DataFrame to a new TSV file
# filtered_df.to_csv("subtitles_en.txt", sep="\t", index=False)
only_movies.to_csv("subtitles_en.txt", sep="\t", index=False)
