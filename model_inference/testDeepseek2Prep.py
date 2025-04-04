"""
Prompting llm through api, measuring response time

Autor: M. Tomasovic
"""

# adjusted example script from: https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free/api
from openai import OpenAI
import json
import time

f = open("key.txt", "r")
api_secret_key = f.read()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_secret_key,
)

filePrompt = r"C:\Users\marti\Music\knn\KNN_project\row_output.txt"

pf = open(filePrompt, "r")
prompt = pf.read()

"""
prepMessages = list(
    {
      "role": "user",
      "content": x
    }
    for x in prompts)

print(prepMessages)
"""
startTime = time.time()
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

resTime = time.time() - startTime
print(f"Time: {resTime}")
#  13.85526442527771 s to response
# with 20s it would take 6 hours to obtain results - need to split into two parts


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

"""
for answer in completion.choices:
    print("Answer:")
    print(answer.message.content)
"""
#print(completion.choices[0].message.content)
