import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "mosaicml/mpt-7b"
tokenizer = AutoTokenizer.from_pretrained(name)

def extractJSON(text):
    filterJSON = re.split(r"[{|}]", text)
    return filterJSON[1]

def extractJSONFromText(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# load json example
with open('example.json', "r", encoding="utf-8") as pf:
    fileContents = json.load(pf)
    # print review
    #print(fileContents["review"])

def encode_labels(genresPredicted, genresAll):
    genre_to_index = {genre: idx for idx, genre in enumerate(genresAll)}
    labels = torch.zeros(len(genresAll))
    for genre in genresAll:
        if genre == genresPredicted:
            labels[genre_to_index[genre]] = 1
    return labels

# load text example
with open('txtexample3.txt', "r", encoding="utf-8") as pf:
    fileContents = pf.read()
    
    # extract JSON from text
    extracted = extractJSONFromText(fileContents)

    # unescape escaped characters - the script works even when no such are in the text
    unescaped = bytes(extracted, "utf-8").decode("unicode_escape")

    # convert to json
    jfileContent = json.loads(unescaped)
    
    # print json parts
    print(jfileContent['review'])
    print(jfileContent['genres'])
    print(jfileContent['rating'])

    # apply Huber loss on rating
    huber = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    
    predictedRating = 2.2
    groundTruthRating = jfileContent['rating']

    pred = torch.tensor(predictedRating)
    gtr = torch.tensor(groundTruthRating)


    print(huber(input=pred, target=gtr))

    # apply cross-entropy loss on review

    compareExampleReview = jfileContent['review'] + " dkvppvs"

    rpred = tokenizer.tokenize(jfileContent['review'],return_tensors="pt")
    rgtr = tokenizer.tokenize(compareExampleReview,return_tensors="pt")
    torch.nn.functional.cross_entropy(input=rpred, target=rgtr)

    # apply Binary Cross-Entropy with sigmoid 
    genresAll = ["Action","Adult","Adventure","Animation","Biography","Comedy","Crime","Documentary","Drama","Family","Fantasy","Film-Noir","Game-Show","History","Horror","Music","Musical","Mystery","News","None","Reality-TV","Romance","Sci-Fi","Short","Sport","Talk-Show","Thriller","War","Western"]
    sigmoid = torch.nn.Sigmoid()
    loss = torch.nn.BCELoss()
    input = torch.randn(3, 2, requires_grad=True)
    target = torch.rand(3, 2, requires_grad=False)
    output = loss(sigmoid(input), target)

    target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
    output = torch.full([10, 64], 1.5)  # A prediction (logit)
    pos_weight = torch.ones([64])  # All weights are equal to 1
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion(output, target)  # -log(sigmoid(1.5))
    # tensor(0.20...)

    # weight and sum