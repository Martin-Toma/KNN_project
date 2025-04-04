This folder contains the scripts associated with creating the dataset.

## Requirements:

- python modules from requirements.txt
- opensubtitles.org dump (Can be aquired from this [RSS source](https://github.com/milahu/opensubtitles-scraper/blob/main/release/opensubtitles.org.dump.torrent.rss) )
- opensubtitles.org [subtitles_all.txt](https://dl.opensubtitles.org/addons/export/) containing the metadata for the subtitles (the dump only contains the id and subtilte files compressed in .zip)
- Imdb metadata title.basics.tsv for the genres and title.ratings.tsv for the average ratings (can be downloaded from [here](https://datasets.imdbws.com/) )
- About 220GB of free space for the subtitle dumps, and temporary files 

## Usage:

1. Put all of the downloaded files into their respective folders, you can use the [folder structure](#Folder-structure):
2. Run main.py (inside this directory)
3. The final database file will be created in the current directory with the name final_final_dataset.db

### Note:

- The script will combine the files into one database file, this can take a long time(about 4h) depending on HW specs
- Because of this the filtered dataset is available from [here](downloads).

## Database structure:

As the dataset container SQLite is used, it's a single file with one table the table structure is as follows:

```sql
CREATE TABLE dataset (
  num INTEGER PRIMARY KEY, -- Subtitle ID
  name TEXT, -- Movie title
  content TEXT -- Subtitle content
  reviews JSON -- Reviews {"rating": 4.5, "reviews": ["This is a review",...]}
  genres TEXT -- 1-3 comma separated genre e.g. "Action, Comedy"
  )
```

## Files:

- main.py: Main script which runs the other scripts in order and loads some of the metadata

- get_reviews.py: Script which downloads the reviews from Amazon

- filter_subtitles_all.py: The metadata for the subtitles is in a file called subtitles_all.txt, this module loads the data from this file and filters it to only include the subtitles which are for movies and in English

- db_merge.py The subtitle dataset consits of multiple database files, this module creates a temporary database file which contains only the subtitles for movies in English

- merge_to_single_table.py: The subtitle dataset is split into multiple tables, with the same column definition but a different name. This module merges all of the tables into one table

- create_sub_dataset.py: Adds a name column to the subtitle dataset, also for some reason the subtitle dataset contains .zip files as subtitles, this module extracts the .zip files and removes (for our purposes useless) unnecessary lines (e.g. "Synced by: ...", or timing information)

- adblocker/opensubtitles_adblocker.py: This module is used to download the subtitles from opensubtitles.org, it uses the RSS feed to get the latest subtitles and downloads them
- merge_subs_reviews_genres_ratings.py: Merges the subtitles, reviews, genres and ratings into the final table

## Folder structure
```
.
├── adblocker
│   ├── opensubtitles_adblocker.py
│   └── source
├── create_sub_dataset.py
├── db_merge.py
├── dir_sturct.txt
├── filter_subtitles_all.py
├── final_final_dataset (Copy).db
├── final_final_dataset.db
├── get_reviews.py
├── imdb_metadata
│   ├── title.basics.tsv
│   └── title.ratings.tsv
├── main.py
├── merge_subs_reviews_genres_ratings.py
├── merge_to_single_table.py
├── opensubtitles_dumps
│   ├── opensubtitles.org.Actually.Open.Edition.2022.07.25
│   │   ├── 404.txt
│   │   └── opensubs.db
│   ├── opensubtitles.org.dump.10000000.to.10099999.v20240820
│   │   └── 100xxxxx.db
│   ├── opensubtitles.org.dump.10100000.to.10199999.v20241003
│   │   └── 101xxxxx.db
│   ├── opensubtitles.org.dump.10200000.to.10299999.v20241124
│   │   └── 102xxxxx.db
│   ├── opensubtitles.org.dump.9180519.to.9521948.by.lang.2023.04.26
│   │   └── langs
│   │       └── eng.db
│   ├── opensubtitles.org.dump.9500000.to.9599999.v20240306
│   │   └── 95xxxxx.db
│   ├── opensubtitles.org.dump.9600000.to.9699999
│   │   ├── 9600xxx.db
│   │   ├── 9601xxx.db
│   │   └── 9699xxx.db # Truncated
│   ├── opensubtitles.org.dump.9700000.to.9799999
│   │   ├── 9700xxx.db
│   │   ├── 9701xxx.db 
│   │   └── 9799xxx.db # Truncated
│   ├── opensubtitles.org.dump.9800000.to.9899999.v20240420
│   │   └── 98xxxxx.db
│   └── opensubtitles.org.dump.9900000.to.9999999.v20240609
│       └── 99xxxxx.db
└── opensubtitles_metadata
    └── subtitles_all.txt
``` 
