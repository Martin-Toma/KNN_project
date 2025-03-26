#!/usr/bin/env python3
import os
import sqlite3
"""
Merges multiple .db files into a single master database (combined.db).
"""
BATCH_SIZE = 5000  # Adjust the batch size as needed

def load_allowed_ids(file_path):
    """
    Reads subtitles_en.txt and returns a set of allowed IDs.
    Assumes the first tab-separated field on each line is the ID.
    """
    allowed_ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                try:
                    allowed_id = int(parts[0])
                    allowed_ids.add(allowed_id)
                except ValueError:
                    print(f"Warning: could not parse id from line: {line}")
    except FileNotFoundError:
        print(f"File {file_path} not found. No filtering will be applied.")
    return allowed_ids

def get_db_files(root_dir):
    """Recursively find all .db files under the given root directory."""
    db_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.db'):
                db_files.append(os.path.join(dirpath, filename))
    return db_files

def get_table_names(conn):
    """Return a list of table names in the given database connection."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]

def get_table_schema(conn, table_name):
    """Retrieve the CREATE TABLE statement for the given table."""
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    result = cursor.fetchone()
    return result[0] if result else None

def merge_databases(master_db_path, db_files, allowed_ids):
    """
    Merge all .db files into the master database.
    For tables named 'subz' or 'zipfiles', only rows with a 'num' value in allowed_ids are inserted.
    Other tables are merged without filtering.
    Processes rows in batches to reduce memory usage.
    """
    master_conn = sqlite3.connect(master_db_path)
    master_cursor = master_conn.cursor()

    for db_file in db_files:
        print(f"Processing {db_file}...")
        src_conn = sqlite3.connect(db_file)
        src_cursor = src_conn.cursor()

        # Get list of tables in the source database.
        tables = get_table_names(src_conn)
        for table in tables:
            # Create the table in the master database if it doesn't exist.
            master_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not master_cursor.fetchone():
                schema = get_table_schema(src_conn, table)
                if schema:
                    print(f"Creating table {table} in master database...")
                    master_conn.execute(schema)
                else:
                    print(f"Warning: No schema found for table {table} in {db_file}. Skipping...")
                    continue

            # Get table column info.
            src_cursor.execute(f"PRAGMA table_info({table})")
            columns_info = src_cursor.fetchall()
            columns = [info[1] for info in columns_info]
            columns_joined = ", ".join(columns)
            placeholders = ", ".join(["?"] * len(columns))

            total_rows = 0
            inserted_rows = 0

            # Decide whether filtering should be applied.
            filter_table = (table in ("subz", "zipfiles")) and allowed_ids and ("num" in columns)

            # If filtering, determine the index for the 'num' column.
            num_index = columns.index("num") if filter_table else None

            # Use a streaming approach with fetchmany to process rows in batches.
            src_cursor.execute(f"SELECT * FROM {table}")
            while True:
                batch = src_cursor.fetchmany(BATCH_SIZE)
                if not batch:
                    break
                total_rows += len(batch)

                if filter_table:
                    filtered = []
                    for row in batch:
                        try:
                            row_id = int(row[num_index])
                        except (ValueError, TypeError):
                            continue
                        if row_id in allowed_ids:
                            filtered.append(row)
                    inserted_rows += len(filtered)
                    if filtered:
                        master_cursor.executemany(
                            f"INSERT OR IGNORE INTO {table} ({columns_joined}) VALUES ({placeholders})",
                            filtered
                        )
                        master_conn.commit()
                else:
                    inserted_rows += len(batch)
                    master_cursor.executemany(
                        f"INSERT OR IGNORE INTO {table} ({columns_joined}) VALUES ({placeholders})",
                        batch
                    )
                    master_conn.commit()
            if filter_table:
                print(f"Inserting {inserted_rows} filtered rows into table {table} (processed {total_rows} rows)...")
            else:
                print(f"Inserting all {inserted_rows} rows into table {table} (no filtering applied)...")
        src_conn.close()
    master_conn.close()
    print("Merging complete!")

if __name__ == '__main__':
    # Set the root directory where your .db files are stored.
    root_directory = '.'
    # Name of the master database file.
    master_database = 'combined.db'
    # Load allowed IDs from subtitles_en.txt.
    allowed_ids = load_allowed_ids("subtitles_en.txt")
    if allowed_ids:
        print(f"Loaded {len(allowed_ids)} allowed IDs from subtitles_en.txt")
    else:
        print("No allowed IDs loaded; all rows will be inserted where applicable.")

    # Get a list of all .db files.
    database_files = get_db_files(root_directory)
    print(f"Found {len(database_files)} database file(s).")
    # Merge them into the master database with filtering and batching.
    merge_databases(master_database, database_files, allowed_ids)

