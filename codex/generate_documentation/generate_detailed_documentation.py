# Access OpenAI Codex API
import argparse
import ast
from datetime import datetime
import json
import os

import openai


# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to generate documentation for a source module
def generate_documentation(prompt, source_code, model="gpt-3.5-turbo-16k"):
    """
    Generates documentation for Python code using an OpenAI LLM model.

    Parameters:
    - prompt (str): The prompt or introduction to the code.
    - source_code (str): The Python source code to be documented.
    - model (str): The model to be used for generating the documentation (default is "gpt-3.5-turbo-16k").

    Returns:
    - documentation_text (str): The generated documentation for the given source code.

    Example Usage:

    >>> prompt = "Generate documentation for this python code."
    >>> source_code = 'print(\"Hello, world!\")'
    >>> documentation = generate_documentation(prompt, source_code)
    >>> print(documentation)

    """
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
    start_time = datetime.now()
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

    documentation_text_list = []

    # create high level module documentation
    module_name = os.path.basename(args.source_file)
    try:
        module_description = generate_documentation(
            "produce a general description of the code and describe what it does",
            source_code
        )

        # initialize list of documentation text with module documentation
        documentation_text_list.append(
            f"# Module:`{module_name}` Overview\n\n{module_description}\n\n"
        )
    except openai.error.InvalidRequestError as e:
        print(f"Error generating module level documentation: {e}")
        documentation_text_list.append(
            f"# Module:`{module_name}` Overview\n\n"
            f"## **Error in generating module level documentation**\n\n"
        )

    # parse source code into abstract syntax tree to pull out lower level details
    tree = ast.parse(source_code)
    for node in tree.body:
        print(node)

        # generate documentation for functions
        if isinstance(node, ast.FunctionDef):
            # extract relevant function information
            function_name = node.name
            function_source = ast.get_source_segment(source_code, node)

            try:
                # generate overview of function
                prompt = f"produce a general description of the function {function_name} and describe what it does"
                documentation_text = generate_documentation(prompt, function_source)
                documentation_text_list.append(f"## Function **`{function_name}`** Overview\n{documentation_text}\n\n")
            except openai.error.InvalidRequestError as e:
                print(f"Error generating function overview documentation: {e}")
                documentation_text_list.append(
                    f"## Function **`{function_name}`** Overview\n"
                    f"### **Error in generating function overview documentation**\n\n"
                )


                # generate detailed description of function
            try:
                prompt = ""
                documentation_text = generate_documentation(prompt, function_source)
                documentation_text_list.append(f"### **Function Details**\n{documentation_text}\n\n")
            except openai.error.InvalidRequestError as e:
                print(f"Error generating function detail documentation: {e}")
                documentation_text_list.append(
                    f"### **Error in generating function detail documentation**\n\n"
                )


        # generate documentation for classes
        elif isinstance(node, ast.ClassDef):
            # extract relevant class information
            class_name = node.name
            class_source = ast.get_source_segment(source_code, node)

            # generate overview of class
            try:
                prompt = f"produce a general description of the class {class_name} and describe what it does"
                documentation_text = generate_documentation(prompt, class_source)
                documentation_text_list.append(f"## Class **`{class_name}`** Overview\n{documentation_text}\n\n")
            except openai.error.InvalidRequestError as e:
                print(f"Error generating class level documentation: {e}")
                documentation_text_list.append(
                    f"# Class **`{class_name}`** Overview\n\n"
                    f"## **Error in generating class level documentation**\n\n"
                )

            # generate documentation for class methods
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    # extract relevant function information
                    method_name = class_node.name
                    print(f"generating doc for method {node.name}.{method_name}()")
                    method_source = ast.get_source_segment(source_code, class_node)

                    # generate overview of function
                    try:
                        prompt = f"produce a general description of the method {method_name} and describe what it does"
                        documentation_text = generate_documentation(prompt, method_source)
                        documentation_text_list.append(f"### Method **`{method_name}`** Overview\n{documentation_text}\n\n")
                    except openai.error.InvalidRequestError as e:
                        print(f"Error generating method overview documentation: {e}")
                        documentation_text_list.append(
                            f"## Function **`{method_name}`** Overview\n"
                            f"### **Error in generating method overview documentation**\n\n"
                        )

                    # generate detailed description of function
                    try:
                        prompt = ""
                        documentation_text = generate_documentation(prompt, method_source)
                        documentation_text_list.append(f"#### **Method Details**\n{documentation_text}\n\n")
                    except openai.error.InvalidRequestError as e:
                        print(f"Error generating method detail documentation: {e}")
                        documentation_text_list.append(
                            f"### **Error in generating method detail documentation**\n\n"
                        )


    # consolidate all the generated documentation text into single markdown file
    documentation_text = "".join(documentation_text_list)

    # output documentation text
    with open(documentation_file, 'w') as f:
        f.write(documentation_text)

    elapsed_time = datetime.now() - start_time

    print(
        f"finished documenting {args.source_file}...created {documentation_file} "
        f"elapsed time {elapsed_time.total_seconds():01.0f} seconds"
    )


if __name__ == '__main__':
    main()






