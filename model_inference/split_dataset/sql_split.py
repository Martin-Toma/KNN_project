"""
Splits dataset to train test and evaluation parts in the SQL tables

Autor: M. Tomasovic
"""

import sqlite3
import random
from tqdm import tqdm

pth = r"C:\Users\marti\Music\knn\KNN_project\final_final_dataset.db"

# connect to the original database
dbConnect = sqlite3.connect(pth)
dbConnect.row_factory = sqlite3.Row  # allow fetching rows as dictionaries
curs = dbConnect.cursor()

# fetch all rows from the original database
curs.execute('''
SELECT
  d.num,
  d.name,
  d.content,
  d.reviews,
  d.genres
FROM dataset d;
''')

rows = [dict(row) for row in curs.fetchall()]  # fetch all rows from the query

# shuffle the rows randomly
random.seed(42)  # seed set
random.shuffle(rows)

# split the data into 1000 for test, 1000 for eval, and the rest for train
test_size = 1000
eval_size = 1000
train_size = len(rows) - test_size - eval_size

test_data = rows[:test_size]
eval_data = rows[test_size:test_size + eval_size]
train_data = rows[test_size + eval_size:]

# function to create tables and insert data into the same database
def create_table_and_insert_data(db_name, table_name, data):
    conn = sqlite3.connect(db_name)  # connect to the same database
    cursor = conn.cursor()

    # create the table for the specific dataset (train, eval, test)
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        num INTEGER,
        name TEXT,
        content TEXT,
        reviews JSON,
        genres TEXT
    );
    ''')

    # insert rows into the table with tqdm progress bar
    with tqdm(total=len(data), desc=f"Inserting into {table_name}", ncols=100) as pbar:
        # insert each row and update the progress bar
        cursor.executemany(f'''
        INSERT INTO {table_name} (num, name, content, reviews, genres)
        VALUES (:num, :name, :content, :reviews, :genres);
        ''', data)
        pbar.update(len(data))  # update progress bar after each insert batch

    # commit changes and close the connection
    conn.commit()
    conn.close()

# insert data into the respective tables - train, eval, test
create_table_and_insert_data(pth, 'train_dataset', train_data)
create_table_and_insert_data(pth, 'eval_dataset', eval_data)
create_table_and_insert_data(pth, 'test_dataset', test_data)

print(f"Train dataset contains {len(train_data)} rows")
print(f"Eval dataset contains {len(eval_data)} rows")
print(f"Test dataset contains {len(test_data)} rows")

# close the connection to the original database
dbConnect.close()
