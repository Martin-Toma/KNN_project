import json
import sqlite3

batch_size = 1000


def get_imdb_id(title, subtitles_en):
    imdb_id = subtitles_en.loc[subtitles_en["MovieName"] == title, "ImdbID"].values
    return int(imdb_id[0]) if len(imdb_id) > 0 else None


def get_rating(imdb_id, ratings):
    rating = ratings.loc[ratings["tconst"] == imdb_id, "averageRating"].values
    return float(rating[0]) if len(rating) > 0 else None


def get_genre(imdb_id, genres):
    genre = genres.loc[genres["tconst"] == imdb_id, "genres"].values
    return genre[0] if len(genre) > 0 else None


def get_review(title: str, reviews):
    return reviews.get(title, None)


def merge_subs_reviews_genres_ratings(db_file, subtitles_en, reviews, ratings, genres):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Create a new table to store the merged data
    cursor.execute("""DROP TABLE IF EXISTS dataset;""")
    cursor.execute(
        """
    CREATE TABLE dataset (num INTEGER PRIMARY KEY, name TEXT UNIQUE, content TEXT,reviews JSON, genres TEXT)
    """
    )
    conn.commit()
    cursor.execute("""SELECT * FROM base_dataset;""")
    insert_cursor = conn.cursor()
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            num = row[0]
            title = row[1]
            content = row[2]
            _reviews = get_review(title, reviews)
            if _reviews is None:  # No reviews for this movie in the dataset
                continue
            imdb_id = get_imdb_id(title, subtitles_en)
            genre = get_genre(imdb_id, genres)
            rating = get_rating(imdb_id, ratings)
            if genre is None or rating is None:
                continue  # No genre or rating for this movie in the dataset
            review = {
                "reviews": _reviews,
                "rating": rating,
            }
            insert_cursor.execute(
                """
            INSERT OR IGNORE INTO dataset (num, name, content, reviews, genres)
            VALUES (?, ?, ?, ?, ?)
            """,
                (num, title, content, json.dumps(review), genre),
            )
    conn.commit()
    # Remove the base dataset if it exists
    cursor.execute(
        """
    DROP TABLE IF EXISTS base_dataset;
    """
    )
    conn.commit()
    # Remove rows where genres was invalid
    cursor.execute("DELETE FROM dataset WHERE genres == '\\N'")
    cursor.execute("DELETE FROM dataset WHERE genres == ''")
    cursor.execute("DELETE FROM dataset WHERE genres is NULL")
    # VACUUM the database
    conn.commit()
    cursor.execute("VACUUM")
    conn.commit()
    cursor.close()
    insert_cursor.close()
    conn.close()
