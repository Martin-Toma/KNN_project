import json
import re
from pathlib import Path

# load db for saving num reference to easily find the review from the db
# prepare db file reference
filePrompt = Path("split_dataset/test_subset.json").resolve()

# load dataset in json format
with open(filePrompt, "r", encoding="utf-8") as pf:
    fileContents = json.load(pf)

# prepare file to store the groupedoutput
groupedFileName = "groupedOutv2.json"
groupedCleanFileName = "groupedOutJustJSONv2.json"
inFolderName = 'responses_v2'

def extractJSON(text):
    filterJSON = re.split(r"[{|}]", text)
    return filterJSON[1]

groupedResponses = []
groupedJustJSONResponses = []

# load responses into a dictionary and add there id of file and id in db
for i, content in enumerate(fileContents):
    try:
        inFile = Path(inFolderName + f"/{i}.txt").resolve()
        print(inFile)
        with open(inFile, 'r', encoding='utf-8') as inFile:
            response = inFile.read()
            groupedResponses.append(
                {"num": content["num"],
                "idx": i,
                "response": response})
            groupedJustJSONResponses.append(
                {"num": content["num"],
                "idx": i,
                "response": '{'+extractJSON(response)+'}'})
    except Exception as e:
        print(e)

with open(groupedFileName, "w", encoding="utf-8") as f:
    json.dump(groupedResponses, f, ensure_ascii=False, indent=4)

with open(groupedCleanFileName, "w", encoding="utf-8") as fJ:
    json.dump(groupedJustJSONResponses, fJ, ensure_ascii=False, indent=4)