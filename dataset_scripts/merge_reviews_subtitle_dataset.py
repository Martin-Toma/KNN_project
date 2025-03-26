import json
import sqlite3

conn = sqlite3.connect('dataset.db')
cursor = conn.cursor()
cursor.execute('Drop table if exists reviews')
cursor.execute('Create table reviews (id integer primary key, title text, review text, rating integer)')
with open('amazon_movies_data.json', 'r') as f:
    js = f.read()
    reviews = json.loads(js)
    del js # free up memory, data is already in reviews
for review in reviews:
    cursor.execute('Insert into reviews (title, review, rating) values (?, ?, ?)', (review['title'], review['cleaned_text'], review['average_rating']))
conn.commit()
cursor.close()
conn.close()