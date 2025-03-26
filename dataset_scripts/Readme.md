This folder contains the scripts associated with creating the dataset.

## Requirements:

- python modules from ../requirements.txt
- opensubtitles.org dump (Can be aquired from [RSS source](https://github.com/milahu/opensubtitles-scraper/blob/main/release/opensubtitles.org.dump.torrent.rss) )
- opensubtitles.org [subtitles_all.txt](https://dl.opensubtitles.org/addons/export/) containing the metadata for the subtitles (the dump only contains the id and subtilte files compressed in .zip)
- About 200GB of free space for the dump files and the final database file

## Usage:

1. Put all of the downloaded files into this folder, the folder structure should look something like [this](#Folder-structure):
2. Run main.py
3. The output is called datasets.db

### Note:

- The script will combine the files into one database file, this can take a long time depending on the amount of files for some of the steps multiprocessing is used, but it can still take over 1hours to complete.
- The download links for subtitle_en.txt and dataset.db can be found [here](downloads).

## Database structure:

- The database which you have downloaded from the note is the output of merge_reviews_subtitle_dataset.py It contains 2 tables:
  - reviews
    - id (autoincrementing private key)
    - title (the name of the movie/series)
    - review (the review text)
    - rating 
  - subtitles
    - num (the number of the subtitle in the dump if you need access to more metadata, you can find it in subtitles_en.txt using:
      `grep '{num}' subtitles_en.txt`)
    - name (the name of the movie/series)
    - content (the subtitle text)

## Files:

### TODO: cleanup (combine the scripts into multiple bigger files, add main.py...)

- download_reviews_dataset.py
- Downloads reviews from Huggingface and saves them to amazon_movies_data.json
- filter_subtitles_all.py
  - From subtitles_all.txt, filters out the subtitles that are not in English and saves them to subtitles_en.txt
- db_merge.py
  - The subtitle dump is split into multiple databases this script takes the databases and merges them into one (also removes subtitles not in subtitles_en.txt)
- merge_to_single_table.py
  - The merged subtitle dump contains multiple tables, with the same structure, this script merges them into one table
- create_sub_dataset.py
  - The subtitle dump contains the subtitles as .zip files, this script extracts the subtitles, removes timing information, adds name column from subtitles_en.txt and creates a new database with the cleaned subtitles
  - TODO: In subtitles_en.txt there are multiple duplicates, also serials have the same name for every episodes
- merge_reviews_subtitle_dataset.py
  - Merges the reviews from amazon_movies_data.json with the subtitle dataset
- create_final.py
  - Combines the subtitle dataset with the reviews into 1 table
  - TODO: Remove duplicates

<!-- Commented out: -->
<!--
## Folder structure
```
.
├── adblocker
│   ├── opensubtitles_adblocker.py
│   └── source
├── create_dataset.py
├── db_merge.py
├── download_reviews_dataset.py
├── main.py
├── merge_to_single_table.py
├── opensubtitles.org.Actually.Open.Edition.2022.07.25
│   ├── 404.txt
│   └── opensubs.db
├── opensubtitles.org.dump.10000000.to.10099999.v20240820
│   ├── 100xxxxx.db
│   └── info
│       └── info.txt
├── opensubtitles.org.dump.10100000.to.10199999.v20241003
│   ├── 101xxxxx.db
│   └── info
│       └── info.txt
├── opensubtitles.org.dump.10200000.to.10299999.v20241124
│   ├── 102xxxxx.db
│   └── info
│       └── info.txt
├── opensubtitles.org.dump.9180519.to.9521948.by.lang.2023.04.26
│   ├── langs
│   │   ├── eng.db
│   ├── missing-404.txt
│   └── missing-dcma.txt
├── opensubtitles.org.dump.9500000.to.9599999.v20240306
│   └── 95xxxxx.db
├── opensubtitles.org.dump.9600000.to.9699999
│   ├── 9600xxx.db
|   |   ... (truncated)
│   └── 9699xxx.db
├── opensubtitles.org.dump.9700000.to.9799999
│   ├── 9700xxx.db
|   |   ... (truncated)
│   └── 9799xxx.db
├── opensubtitles.org.dump.9800000.to.9899999.v20240420
│   └── 98xxxxx.db
├── opensubtitles.org.dump.9900000.to.9999999.v20240609
│   ├── 99xxxxx.db
│   └── info
│       ├── info.txt
│       └── magnets.txt
├── Readme.md
├── subtitle_dataset_source
└── subtitles_all.txt
``` -->
