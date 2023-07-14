# Access OpenAI Codex API
import argparse
import json
import os

import openai

# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to translate SAS code to Python
def translate_sas_to_python(prompt, sas_code):
    completion_prompt = f'{prompt}:\n\n{sas_code}\n\nPython code:'

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",   #'text-davinci-003',
        messages=[{"role": "user", "content": completion_prompt}],
        max_tokens=2048,
        temperature=0.0,
        n=1,
        stop=None,
    )

    python_code = response.choices[0].message.content.strip()
    return python_code

if __name__ == '__main__':
    # retrieve sas file name from first parameter in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("sas_file", help="SAS file to convert to Python")

    # retrieve output directory from second parameter in command line
    parser.add_argument("output_dir", help="Directory to write Python file to")

    args = parser.parse_args()

    # read in sas file
    with open(args.sas_file, 'r') as f:
        sas_code = f.read()

    print(f"starting conversion for {args.sas_file}...")
    # translate sas code to python
    prompt = "convert this SAS program to Python"
    python_code = translate_sas_to_python(prompt, sas_code)

    # write python code to file
    # get file name from sas file name
    file_name = os.path.basename(args.sas_file)

    # create file name for python file
    python_file = os.path.join(args.output_dir, file_name.replace('.sas', '.py'))
    with open(python_file, 'w') as f:
        f.write(python_code)

    print(f"finished conversion for {args.sas_file}...created {python_file}")




