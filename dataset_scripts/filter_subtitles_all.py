import pandas as pd

"""This module removes non english and non movie subtitle metadata from subtitles_all.txt and saves the results as subtitles_en.txt
    The reason why we have to remove tv shows is that the reviews dataset doesn't differentiate between episodes of TV shows or even seasons."""


def filter_subtitles_all(subtitles_all):
    # Read the TSV file and skip lines that don't have the expected number of columns
    df = pd.read_csv(subtitles_all, sep="\t", dtype=str, on_bad_lines="skip")

    # Filter rows: keep only rows where ISO639 (lowercased) is either "en" or "eng"
    filtered_df = df[df["ISO639"].str.lower().isin(["en", "eng"])]
    # Remove serials uncategorized movies (the reviews dataset doesn't differentiate between episodes of TV shows)
    only_movies = filtered_df.query("MovieKind == 'movie'")
    ## return result
    return only_movies
