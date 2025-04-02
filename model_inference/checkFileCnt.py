import os
from pathlib import Path

folder = Path("responses_v2").resolve()
max_file_index = 999

def find_missing_files(folder_path, max_index):
    expected_files = {f"{i}.txt" for i in range(max_index + 1)}
    existing_files = set(os.listdir(folder_path))
    
    missing_files = expected_files - existing_files
    
    if missing_files:
        print("Missing files:")
        for file in sorted(missing_files, key=lambda x: int(x.split('.')[0])):
            print(file)
    else:
        print("No files are missing.")

find_missing_files(folder, max_file_index)
