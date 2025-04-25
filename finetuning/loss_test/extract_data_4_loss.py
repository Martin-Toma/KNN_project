import json
import re

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