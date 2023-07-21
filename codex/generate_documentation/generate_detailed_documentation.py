# Access OpenAI Codex API
import argparse
import ast
from collections import namedtuple
import json
import os

import openai

DocumentationItem = namedtuple("DocumentationItem", ["name", "doc_type", "documentation"])

# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to generate documentation for a source module
def generate_documentation(prompt, source_code, model="gpt-3.5-turbo-16k"):
    completion_prompt = f'{prompt}:\n\n{source_code}\n\nPython code:'

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": completion_prompt}],
        max_tokens=8096,
        temperature=0.0,
        n=1,
        stop=None,
    )

    documentation_text = response.choices[0].message.content.strip()
    return documentation_text


# main function
def main():
    # retrieve source file name from first parameter in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Source file to document")

    # retrieve documentation file from second parameter in command line
    parser.add_argument("documentation_file", help="file to contain the documentation")

    args = parser.parse_args()

    # read in source file
    with open(args.source_file, 'r') as f:
        source_code = f.read()
    print(f"starting to document {args.source_file}")

    # create file name for python file
    documentation_file = args.documentation_file

    # create high level module documentation
    module_name = os.path.basename(args.source_file)
    module_description = generate_documentation(
        "produce a general description of the code and describe what it does",
        source_code
    )

    # initialize list of documentation text with module documentation
    documentation_text_list = [
        f"# Module:`{module_name}` Overview\n\n{module_description}\n\n"
    ]

    # parse source code into abstract syntax tree to pull out lower level details
    tree = ast.parse(source_code)
    for node in tree.body:
        print(node)

        # generate documentation for functions
        if isinstance(node, ast.FunctionDef):
            # extract relevation function information
            function_name = node.name
            function_source = ast.get_source_segment(source_code, node)

            # generate overview of function
            prompt = f"produce a general description of the function {function_name} and describe what it does"
            documentation_text = generate_documentation(prompt, function_source)
            documentation_text_list.append(f"## Function **`{function_name}`** Overview\n{documentation_text}\n\n")

            # generate detailed description of function
            prompt = ""
            documentation_text = generate_documentation(prompt, function_source)
            documentation_text_list.append(f"### **Function Details**\n{documentation_text}\n\n")

    # consolidate all the generated documentation text into single markdown file
    documentation_text = "".join(documentation_text_list)

    # output documentation text
    with open(documentation_file, 'w') as f:
        f.write(documentation_text)

    print(f"finished documenting {args.source_file}...created {documentation_file}")


if __name__ == '__main__':
    main()






