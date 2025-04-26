"""
The task of creating review, rating and selecting geners from subtitles has one input and 
one output, so instruction format is 
used https://huggingface.co/docs/trl/sft_trainer#dataset-format-support
This script reformats the dataset splits to the instructio format, which can be fed directly
to the SFTTrainer.
Also it only takes random 5 samples of reviews so it crated balanced dataset while preserves 
diversity.

Autor: M. Tomasovic
"""
import json
import re
import random

splits = ['train_dataset', 'eval_dataset', 'test_dataset'] # all dataset splits

def extractJSONFromText(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def reformat_genres(genres):
    genresLst = []
    if genres is None:
        return json.dumps(["None"])
    if genres.find(',') != -1:
        genresLst = genres.split(',') # make list of genres
    else:
        genresLst.append(genres)
    #escapedGenresLst = (str(genresLst)).replace("'", "\\'") # escape '
    
    return json.dumps(genresLst)

# format dataset to {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
def instruction_reformat_dataset(movieData):
    # the prompt contains 'content' - the subtitles
    subtitles = movieData['content']
    # reviews contains 'rating' and multiple 'reviews' which are separated to multiple training pairs
    # add 'genres' to the completion part
    
    # extract JSON from text
    extracted = extractJSONFromText(movieData['reviews'])

    # unescape escaped characters - the script works even when no such are in the text
    unescaped = bytes(extracted, "utf-8").decode("unicode_escape")
    
    # review part
    reviewPart = json.loads(unescaped)

    instructions = [] # prepare list to store instruction for current movie
    
    # extract rating and genres which are same for a movie
    rating = reviewPart['rating'] 
    genres = movieData['genres']
    genres = reformat_genres(genres)

    # movie has more reviews
    randomReviews = []
    if (len(reviewPart['reviews']) >= 5):
        # take random 5
        randomReviews = random.sample(reviewPart['reviews'], 5)
    else:
        # when less than 5 samples take them all
        randomReviews = reviewPart['reviews']
    
    for review in randomReviews:
        generated = "{  \n  \"rating\": " + str(rating) + ",  \n  \"genres\": " + genres + ",  \n  \"review\": \"" + str(review) + "\"  \n}"
        instruction = "{\"prompt\": \"" + subtitles +"\", \"completion\": " + generated + "}"
        instructions.append(instruction)
    
    return instructions


for split in splits:

    # load dataset in json format
    splitJsonPth = split  + ".json"                 # load path
    splitNewJsonPth = "instruction_" + splitJsonPth # store path
    with open(splitJsonPth, "r", encoding="utf-8") as pf:
        allData = json.load(pf)

        # create prompts for each movie max 5 with different reviews
        dset = []
        for movie in allData:
            # need to add multiple review examples per movie
            dset.extend(instruction_reformat_dataset(movie))

        # store the result
        with open(splitNewJsonPth, "w", encoding="utf-8") as wf:
            json.dump(dset, wf, ensure_ascii=False, indent=4)
        