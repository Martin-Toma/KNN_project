# adjusted example script from: https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free/api
from openai import OpenAI
import json

f = open("key.txt", "r")
api_secret_key = f.read()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_secret_key,
)

filePrompt = r"C:\Users\marti\Music\knn\separate project\prompts.json"

pf = open(filePrompt, "r")
prompts = json.load(pf)

"""
prepMessages = list(
    {
      "role": "user",
      "content": x
    }
    for x in prompts)

print(prepMessages)
"""

for prompt in prompts:
  completion = client.chat.completions.create(
    extra_body={},
    model="deepseek/deepseek-chat-v3-0324:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }]
  )
  print(completion.choices[0].message.content)

"""
for answer in completion.choices:
    print("Answer:")
    print(answer.message.content)
"""
#print(completion.choices[0].message.content)