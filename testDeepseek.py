# adjusted example script from: https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free/api
from openai import OpenAI

f = open("key.txt", "r")
api_secret_key = f.read()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_secret_key,
)

completion = client.chat.completions.create(
  extra_body={},
  model="deepseek/deepseek-chat-v3-0324:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)