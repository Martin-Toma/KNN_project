#!/usr/bin/env python3
import sqlite3
import os

"""
Merges 'subz' and 'zipfiles' tables from combined.db into a single 'subs' table in combined2.db.
"""


def create_subs_table(dst_conn):
    """
    Create the single 'subs' table in the destination database.
    """
    dst_conn.execute("DROP TABLE IF EXISTS subs;")
    dst_conn.execute(
        """
        CREATE TABLE subs (
            num INTEGER PRIMARY KEY,
            name TEXT,
            file BLOB
        );
    """
    )
    dst_conn.commit()


def copy_from_subz(src_conn, dst_conn):
    """
    Copy rows from 'subz' in combined.db into 'subs' in combined2.db.
    """
    # subz columns: (num INTEGER PK, name TEXT, file BLOB)
    src_cursor = src_conn.cursor()
    rows = src_cursor.execute("SELECT num, name, file FROM subz").fetchall()

    dst_cursor = dst_conn.cursor()
    dst_cursor.executemany("INSERT INTO subs (num, name, file) VALUES (?, ?, ?);", rows)
    dst_conn.commit()


def copy_from_zipfiles(src_conn, dst_conn):
    """
    Copy rows from 'zipfiles' in combined.db into 'subs' in combined2.db.
    """
    # zipfiles columns: (num INTEGER PK, name TEXT, content BLOB)
    # We want to map 'content' -> 'file'
    src_cursor = src_conn.cursor()
    rows = src_cursor.execute("SELECT num, name, content FROM zipfiles").fetchall()

    dst_cursor = dst_conn.cursor()
    dst_cursor.executemany("INSERT INTO subs (num, name, file) VALUES (?, ?, ?);", rows)
    dst_conn.commit()


def merge_tables(src_db_path, dst_db_path):
    # Connect to both databases
    src_conn = sqlite3.connect(src_db_path)
    dst_conn = sqlite3.connect(dst_db_path)

    # 1) Create the 'subs' table in combined2.db
    create_subs_table(dst_conn)

    # 2) Copy rows from 'subz'
    copy_from_subz(src_conn, dst_conn)

    # 3) Copy rows from 'zipfiles'
    copy_from_zipfiles(src_conn, dst_conn)

    # Close connections
    src_conn.close()
    dst_conn.close()

    print("Done! Created combined2.db with a single 'subs' table.")


def main():
    # Paths to the existing and the new database
    src_db_path = "combined.db"  # existing DB (has subz, zipfiles)
    dst_db_path = "combined2.db"  # new DB (will have only subs)

    # Remove combined2.db if it already exists (optional)
    if os.path.exists(dst_db_path):
        os.remove(dst_db_path)

    # Connect to both databases
    src_conn = sqlite3.connect(src_db_path)
    dst_conn = sqlite3.connect(dst_db_path)

    # 1) Create the 'subs' table in combined2.db
    create_subs_table(dst_conn)

    # 2) Copy rows from 'subz'
    copy_from_subz(src_conn, dst_conn)

    # 3) Copy rows from 'zipfiles'
    copy_from_zipfiles(src_conn, dst_conn)

    # Close connections
    src_conn.close()
    dst_conn.close()

    print("Done! Created combined2.db with a single 'subs' table.")


if __name__ == "__main__":
    main()
