"""
Extracts all three subsets from SQL databes to separate JSON files

Autor: M. Tomasovic
"""
import sqlite3
import json

splits = ['train_dataset', 'eval_dataset', 'test_dataset']

# set the path to database
pth = r"C:\Users\marti\Music\knn\KNN_project\final_final_dataset.db"

# connect to the SQLite database
dbConnect = sqlite3.connect(pth)
dbConnect.row_factory = sqlite3.Row  # allow fetching rows as dictionaries
curs = dbConnect.cursor()

# iterate over splits
for d in splits:
    outputFileName = d

    # extract a random subset of rows including the genres column
    curs.execute(f'''
    SELECT
    d.num,
    d.name,
    d.content,
    d.reviews,
    d.genres
    FROM {d} d
    ''')

    # fetch all rows and convert each row to a dictionary
    rows = [dict(row) for row in curs.fetchall()]

    # save the rows to a JSON file
    with open(outputFileName+".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    print(f"Saved test subset with genres to {outputFileName}")

# close the database connection
dbConnect.close()