# KNN_project

The project aims to fine-tune a language model for generating movie reviews and estimating movie genres. This model can be used to review new movies, assist in improving movie scripts, and help viewers decide whether to watch a film. 

## Model baseline part

Explained model part folders and files
- `.gitignore`
- `LICENSE`
- `meeting3.txt`
- `notes.md`
- `README.md`
- `eval/`
  - `llm_perplexity.py`: perplexity calculation for huggingface models
- `local_model/`: scripts to experiment with little hugging face models
  - `run.py`: run inference of small 3B llm model, preparation for bigger one
  - `tokenCnt.py`: estimates the number of 
  tokens per movie subtitle
  - `prompts.json`: input for run.py
- `model_inference/`: contains files for baseline
  - `checkFileCnt.py`: after inference the result is stored in multiple files this checks the file count 
  - `datasetStats.py`: for the dataset stored in SQL database, plots statistics
  - `FilesMap.md`: file containg map of files
  - `groupedOut.json`: contains all responses of deepseek on test dataset
  - `groupFiles.py`: the output of testDeepseek.py is in multiple files, this script processes them and creates two JSON files, groupedOut/groupedOutv2.json contains whole DeepSeek response, groupedOutJustJSON/groupedOutJustJSONv2.json contains just the required triplet JSON data.
  - `README.md`: contains description of database manipulation
  - `perplexity_stats.py`: perplexity statistics across the whole test dataset
  - `sotaModelInfo.md`: information about possible and the chosen llm for baseline
  - `testDeepseek.py`: through api sends prompts to DeepSeek V3 model and stores repsonses and perplexity for each movie
  - `testDeepseek2.py`: prompting llm through api, measuring response time
  - `testDeepseek.py`: testing cyclic api prompting
  - `trim_dset_part.py`: trims inner part of the subtitles
  - `split_dataset/`: scripts to split the dataset in sql
    - `sql_split.py`: splits dataset to train test and evaluation parts in the SQL tables
    - `subset2JSON.py`: extracts subset from SQL databes to JSON file
    - `test_subset.json`: the product of `subset2JSON.py`
    - `oldway/`
      - `extractTestSubset.py`: extraction of test subset from the whole database
- `reviews_dataset/`: script to extract parts of huggingface amazon review dataset
  - `datase_links.md`: link for original dataset
  - `prepareRewiews.py`: script to prepare the dataset - name, review, rating columns
- `whole_dataset_stats/`: the whole dataset stats scripts
  - `genres_stats.png`: plots statistics about geners
  - `reviewStats.png`: plots statistics about reviews

#### Test subset extraction

Now the test subset extraction is done by extracting test split using `subset2JSON.py`. The dataset is divided into 1,000 samples for testing, 1,000 for evaluation, and approximately 31,000 for training. The link for the splitted dataset: [splitDataset](https://akirabox.com/9QWmpZ7XzEB6/file). Old way was: (In folder split_dataset, there is a script `extractTestSubset.py`, which extracts exactly 1000 random samples from dataset `final_dataset2.db` and saves it in JSON format to `test_subset.json` file.)

#### Responses from Deepseek V3
In file `testDeepsee.py`, test dataset is send prompt by prompt to DeepSeek V3 model via api. The responses of the model are stored in folder `responses` or `responses2` each response in it's own file named by the id - the order in the test dataset. 

#### Process the responses
The responses from `responses` folder are processed using script `groupFiles.py`. The script outputs two JSON files:
1. `groupedOut.json` containes whole model responses in format: 
```
    [
        {
        "num": content["num"],
        "idx": i,
        "response": response
        },
        ...
    ]
```

2. `groupedOutJustJSON.json` containes part of model responses which has JSON format `{"rating": <rating>, "review": <review_text>}` and these parts of responses are stored under `response` in JSON file in following format: 
```
    [
        {
            "num": content["num"],
            "idx": i,
            "response": '{'+extractJSON(response)+'}'
        },
        ...
    ]
```

Problem with 295-th sample too large 159259 length. Solved in `trim_dset_part.py`.
