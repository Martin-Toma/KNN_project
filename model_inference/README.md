## Test subset extraction

In folder split_dataset, there is a script `extractTestSubset.py`, which extracts exactly 1000 random samples from dataset `final_dataset2.db` and saves it in JSON format to `test_subset.json` file.

## Responses from Deepseek V3
In file `testDeepsee.py`, test dataset is send prompt by prompt to DeepSeek V3 model via api. The responses of the model are stored in folder `responses` each response in it's own file named by the id - the order in the test dataset. 

## Process the responses
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