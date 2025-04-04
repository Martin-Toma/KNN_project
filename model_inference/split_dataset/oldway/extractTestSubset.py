"""
Extraction of test subset from the whole database

Author: M. Tomasovic
"""

import sqlite3
import json

pth = r"C:\Users\marti\Music\knn\KNN_project\final_final_dataset.db"
outputFileName = "test_subset.json"

dbConnect = sqlite3.connect(pth)
dbConnect.row_factory = sqlite3.Row  # allow fetching rows as dictionaries
curs = dbConnect.cursor()
# extract 1000 random rows
curs.execute('''
SELECT
  d.num,
  d.name,
  d.content,
  d.reviews
FROM dataset d
ORDER BY RANDOM()
LIMIT 1000;
''')

rows = [dict(row) for row in curs.fetchall()]  # fetch all rows from query

# convert list of json into a single json file
with open(outputFileName, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=4)

print(f"Saved to {outputFileName}")

dbConnect.close() # close db connection