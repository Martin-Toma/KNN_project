import pandas as pd
from filter_subtitles_all import filter_subtitles_all as fsm
from db_merge import combine_dbs as cdb
from merge_to_single_table import merge_tables as mts
from create_sub_dataset import create_base_dataset as cbd
from get_reviews import get_reviews as gr
from merge_subs_reviews_genres_ratings import merge_subs_reviews_genres_ratings as msrgr
import os
import concurrent.futures

## Locations

### Inputs
subtitles_all = os.path.join("opensubtitles_metadata", "subtitles_all.txt")

subtitles_db_dumps = os.path.join("opensubtitles_dumps")

imdb_basics = os.path.join("imdb_metadata", "title.basics.tsv")
imdb_ratings = os.path.join("imdb_metadata", "title.ratings.tsv")

##Outputs
### Temporary files
os.makedirs("tmp", exist_ok=True)
combined_db = os.path.join("tmp", "combined.db")
combined_db2 = os.path.join("tmp", "combined2.db")
### Final files
result = "final_final_dataset.db"


def load_ratings():
    rating_df = pd.read_csv(imdb_ratings, sep="\t", dtype=str)
    rating_df["tconst"] = rating_df["tconst"].str.replace("tt", "", regex=False)
    rating_df["tconst"] = pd.to_numeric(rating_df["tconst"], errors="coerce")
    rating_df = rating_df.dropna(subset=["tconst"])
    rating_df["tconst"] = rating_df["tconst"].astype(int)
    return rating_df


def load_genres():
    genres_df = pd.read_csv(imdb_basics, sep="\t", dtype=str)
    genres_df["tconst"] = genres_df["tconst"].str.replace("tt", "", regex=False)
    genres_df["tconst"] = pd.to_numeric(genres_df["tconst"], errors="coerce")
    genres_df = genres_df.dropna(subset=["tconst"])
    genres_df["tconst"] = genres_df["tconst"].astype(int)
    return genres_df


def main():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # During database operations, we can load the reviews, ratings and genres in parallel
        future_reviews = executor.submit(gr)
        future_ratings = executor.submit(load_ratings)
        future_genres = executor.submit(load_genres)
        # Create a DataFrame from the subtitles_all file, with only movies and english subtitles
        subtitles_en = fsm(subtitles_all)
        # Combine the subtitles database dumps into a single database (also filter out rows not in the subtitles_en DataFrame)
        cdb(subtitles_db_dumps, combined_db, subtitles_en)
        # # The single database contains 2 tables, this is due to how the subtitle database dumps were created. Let's merge them into a single table TODO: UNCOMMENT THIS LINE
        mts(combined_db, combined_db2)
        # Some of the subtitles contain ads, also for some reason they are saved as .zip files. The next function unpacks the .zip files, removes the ads and other (for our purposes) unnecessary metadata (e.g. timestamps)
        cbd(combined_db2, result, subtitles_en)
        # Wait for all futures to complete and get the results
        ratings = future_ratings.result()
        genres = future_genres.result()
        reviews = future_reviews.result()
        # Merge everything into a single database
        msrgr(result, subtitles_en, reviews, ratings, genres)
        # Remove the temporary files
        os.remove(combined_db)
        os.remove(combined_db2)
        # remove tmp directory
        os.rmdir("tmp")
        print("Done! Created final_final_dataset.db.")

if __name__ == "__main__":
    # Run the main function
    main()
