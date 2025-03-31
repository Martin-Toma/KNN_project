# adjusted example script from: https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free/api
from openai import OpenAI
import json
from pathlib import Path

# load key to connect to client
f = open("key.txt", "r")
api_secret_key = f.read()

# prepare client
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_secret_key,
)

# prepare file reference
filePrompt = Path("split_dataset/test_subset.json").resolve()
# prepare folder to save response reference
outFolder = 'responses'

# load dataset in json format
with open(filePrompt, "r", encoding="utf-8") as pf:
    fileContents = json.load(pf)

# prepare command
command = 'Please provide a review in the following format: {"rating": <rating>, "review": <review_text>}. Make sure the rating is a number in range 0.0 to 5.0 and the review consists of sentences summarizing, describing and evaluating the movie. Here are subtitles of the movie: '

for idx, fileContent in enumerate(fileContents):
  if idx > 178:
    try:
      prompt = command + fileContent["content"]
      
      completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
            "role": "user",
            "content": prompt
            }],
        temperature=0.0,  # make responses more consistent
      )

      modelResponse = completion.choices[0].message.content

      # store it to a file
      outFile = Path(outFolder + f"/{idx}.txt").resolve()
      with open(outFile, 'w', encoding='utf-8') as outf:
        outf.write(modelResponse)
    except Exception as e:
      print(idx)
      print(e)

    