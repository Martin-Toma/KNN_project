import csv
import sqlite3
import zipfile
import io
import os
import re
import pysrt
import chardet
from adblocker.opensubtitles_adblocker import OpensubtitlesAdblocker as adb
import multiprocessing

adb_instance = adb()
batch_size = 50
worker_count = 10 #int(multiprocessing.cpu_count()/2)


def parse_subs(subs, names):
    rows = []
    for sub in subs:
        if str(sub[0]) in names:
            # Extract sub[2] (zip) and read .srt file from zip
            files = unzip_in_memory(sub[2])
            subtitle = b""
            sub_type = 0  # 0: .srt, 1: .sub
            for file_name, file_content in files.items():
                if file_name.endswith(".srt"):
                    subtitle += (
                        b"\n" + file_content
                    )  # Multiple subtitle files in one zip e.g. for a 2 DVD movie
                if file_name.endswith(".sub"):
                    sub_type = 1
                    subtitle += b"\n" + file_content
            # Remove ads
            text = ""
            if sub_type == 0:
                subtitle = adb_instance.filter_subtitle_bytes(subtitle)
                text = parse_srt_bytes(subtitle)
                text = text.strip()
            else:
                subtitle = adb_instance.filter_subtitle_bytes(subtitle).decode(
                    "utf-8", errors="replace"
                )
                text = clean_sub(subtitle)
            rows.append((sub[0], names[str(sub[0])], text))
    return rows

def parse_srt_bytes(srt_bytes):
    # Detect encoding
    try:
        detected = chardet.detect(srt_bytes)
        encoding = detected["encoding"] or "utf-8"

    # Decode the bytes to string
        srt_text = srt_bytes.decode(encoding, errors="replace")
    except Exception as e:
        print(e)
        srt_text = srt_bytes.decode("utf-8", errors="replace")
    # Parse subtitles
    subs = pysrt.from_string(srt_text)

    # Extract only text from subtitles
    return "\n".join([sub.text for sub in subs])


def clean_sub(subtitle):
    """
    Cleans .sub subtitle text by:
    - Removing timing markers {xxxx}{yyyy}
    - Splitting lines at '|'
    - Normalizing spaces
    """
    subtitle = re.sub(r"\{\d+\}\{\d+\}", "", subtitle)  # Remove {xxxx}{yyyy}
    subtitle = subtitle.replace("|", "\n")  # Split at '|'
    subtitle = re.sub(r"\s+", " ", subtitle).strip()  # Normalize spaces
    return subtitle


def unzip_in_memory(binary_data):
    """
    Unzips a binary ZIP file in memory and returns a dictionary
    where keys are file names and values are extracted file contents (bytes).
    """
    extracted_files = {}

    # Open the zip file from memory
    with zipfile.ZipFile(io.BytesIO(binary_data), "r") as zip_ref:
        for file_name in zip_ref.namelist():
            with zip_ref.open(file_name) as file:
                extracted_files[file_name] = file.read()  # Read file into memory

    return extracted_files


def create_base_dataset(src_db, dst_db, subtitles_en):
    conn = sqlite3.connect(dst_db)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS base_dataset;")
    cursor.execute(
        " CREATE TABLE base_dataset (num INTEGER PRIMARY KEY, name TEXT, content BLOB)"
    )
    conn2 = sqlite3.connect(src_db)
    cursor2 = conn2.cursor()
    names = dict(zip(subtitles_en["IDSubtitle"], subtitles_en["MovieName"]))
    cursor2.execute("SELECT * FROM subs")
    # This could be more efficient but this code is only ran once
    with multiprocessing.Pool(worker_count) as pool:
        while True:
            sublist = []
            subs = cursor2.fetchmany(batch_size * worker_count)
            if not subs:
                break
            # Split subs into chunks for parallel processing
            for i in range(0, len(subs), batch_size):
                sublist.append(subs[i : i + batch_size])
            if not sublist:
                break
            # rows = parse_subs(subs,names)
            results = pool.starmap(parse_subs, [(subs, names) for subs in sublist])
            for rows in results:
                cursor.executemany(
                    "INSERT OR IGNORE INTO base_dataset VALUES (?, ?, ?)", rows
                )
                conn.commit()
            results = None
    # Remove rows where the subtitles were not exported
    cursor.execute("DELETE FROM base_dataset WHERE content IS NULL OR content = '';")
    cursor.execute("DELETE from base_dataset where name = 'Empty Movie (SubScene)';")
    conn.commit()
    conn.close()
    conn2.close()


# if __name__ == "__main__":
#     if os.path.exists(dataset_file):
#         os.remove(dataset_file)
#     create_dataset()
