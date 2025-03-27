import json
import sqlite3

conn = sqlite3.connect('dataset.db')
cursor = conn.cursor()
cursor.execute('''SELECT
  d.num,
  d.name,
  d.content,
  json_group_array(
    json_object(
      'text', r.review,
      'rating', r.rating
    )
  ) AS reviews
FROM dataset d
INNER JOIN reviews r ON d.name = r.title
GROUP BY d.num, d.name, d.content;
''')
rows = cursor.fetchall()
cursor.close()
conn.close()
conn = sqlite3.connect('final_dataset.db')
cursor = conn.cursor()
cursor.execute('Drop table if exists dataset')
cursor.execute('Create table dataset (num integer primary key, name text UNIQUE, content blob, reviews JSON)')
for row in rows:
    cursor.execute('Insert OR IGNORE into dataset (num, name, content, reviews) values (?, ?, ?, ?)', row)
conn.commit()
