# Access OpenAI Codex API
import argparse
import json
import os

import openai
import tiktoken

# CONSTANTS
THIS_LLM = "gpt-3.5-turbo-16k"


# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to generate documentation for a source module
def generate_documentation(prompt, source_code):
    """
    Generates Python documentation for the given prompt and source code.

    Args:
        prompt (str): The description or prompt for generating the documentation.
        source_code (str): The Python source code to be documented.

    Returns:
        str: The generated Python documentation.

    Raises:
        OpenAIError: If there is an error in the OpenAI API request.

    Examples:
        >>> prompt = "Generate Python documentation for a function"
        >>> source_code = "def greet(name):\n    '''This function greets the given name.'''\n    print('Hello,', name)"
        >>> documentation = generate_documentation(prompt, source_code)
        >>> print(documentation)
    """

    completion_prompt = f'{prompt}:\n\n{source_code}\n\nPython code:'

    # compute token size
    encoding = tiktoken.encoding_for_model(THIS_LLM)
    token_size = len(encoding.encode(completion_prompt))
    print(f">>>model {THIS_LLM}, encoded source file token size: {token_size}")

    response = openai.ChatCompletion.create(
        model=THIS_LLM,
        messages=[{"role": "user", "content": completion_prompt}],
        max_tokens=8096,
        temperature=0.0,
        n=1,
        stop=None,
    )

    documentation_text = response.choices[0].message.content.strip()
    return documentation_text

if __name__ == '__main__':
    # retrieve source file name from first parameter in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Source file to document")

    # retrieve documentation file from second parameter in command line
    parser.add_argument("documentation_file", help="file to contain the documentation")

    #retrieve prompt from third parameter in command line, with default value
    parser.add_argument(
        "--prompt",
        help="Prompt to use to generate documentation",
        default="produce a general description of the code and describe what it does")

    args = parser.parse_args()

    # read in sas file
    with open(args.source_file, 'r') as f:
        source_file = f.read()

    # generate documentation
    print(f"starting to document {args.source_file}...with prompt '{args.prompt}'")
    # translate sas code to python
    prompt = args.prompt
    documentation_text = generate_documentation(prompt, source_file)

    # create file name for python file
    documentation_file = args.documentation_file
    with open(documentation_file, 'w') as f:
        f.write(documentation_text)

    print(f"finished documenting {args.source_file}...created {documentation_file}")




