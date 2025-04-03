# adjusted example script from: https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free/api
from openai import OpenAI
import json
from pathlib import Path
import math

# load key to connect to client
f = open("key.txt", "r")
api_secret_key = f.read()

# prepare client
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_secret_key,
)

# prepare file reference
filePrompt = Path("split_dataset/test_subset_v2_trimmed.json").resolve()
# prepare folder to save response reference
outFolder = 'responses_v2'
outFolderPPL = 'perplexities'

# load dataset in json format
with open(filePrompt, "r", encoding="utf-8") as pf:
    fileContents = json.load(pf)

# calculate perplexity
def get_perplexity(logprobs):
  n = len(logprobs)
  if n == 0:
    return 0
  return math.exp(-sum(logprobs) / n)

# prepare command
command = 'Please provide a review in the following format: {"rating": <rating>, "genres": <genres>, "review": <review_text>}. Make sure the rating is a number in the range of 1.0 to 10.0. In the genres field, include 1-3 values from [Action, Adult, Adventure, Animation, Biography, Comedy, Crime, Documentary, Drama, Family, Fantasy, Film-Noir, Game-Show, History, Horror, Music, Musical, Mystery, News, None, Reality-TV, Romance, Sci-Fi, Short, Sport, Talk-Show, Thriller, War, Western]. The review should evaluate the movie in a few sentences. Here are the movie subtitles: '

for idx, fileContent in enumerate(fileContents):
  print(idx)
  if idx >= 739 and idx < 800:
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
        logprobs=True
      )

      modelResponse = completion.choices[0].message.content
      # store response to a file
      outFile = Path(outFolder + f"/{idx}.txt").resolve()
      
      with open(outFile, 'w', encoding='utf-8') as outf:
        outf.write(modelResponse)

      # calculate perplexity from log probabilities
      log_probs = [token.logprob for token in completion.choices[0].logprobs.content]
      ppl = get_perplexity(log_probs)

      # store perplexity to a file
      outFile = Path(outFolderPPL + f"/{idx}.txt").resolve()
      with open(outFile, 'w', encoding='utf-8') as outf:
        outf.write(str(ppl))

    except Exception as e:
      print(idx)
      print(e)
      print(completion.choices)

    