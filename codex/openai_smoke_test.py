# import os
import json
import openai

# print version of openai
print(f"openai.__version__: {openai.__version__}")

# openai.api_key = os.getenv("OPENAI_API_KEY")
# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": "What is the capital of Hawaii?\n"
    },
    ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

# print the response from query
print(f"response.choices[0].message.content: {response.choices[0].message.content}")

# full response
print(f"\nresponse: {response}")
