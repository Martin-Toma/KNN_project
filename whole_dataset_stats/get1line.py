import sqlite3

pth = "C:\\Users\\marti\\Music\\knn\\separate project\\datastest\\dataset.db"

conn = sqlite3.connect(pth)
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
GROUP BY d.num, d.name, d.content limit 1;
''')

# Fetch just the first row
single_row = cursor.fetchone()  # Returns the first row or None if empty
cursor.close()
conn.close()

if single_row:
    with open("row_output.txt", "w", encoding="utf-8") as f:
        f.write(str(single_row))
    print("Full row saved to 'row_output.txt'")