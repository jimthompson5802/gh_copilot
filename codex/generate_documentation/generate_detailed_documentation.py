# Access OpenAI Codex API
import argparse
import ast
from datetime import datetime
import json
import os

import openai

# Prompt constants
PROMPT_DESCRIPTION_PREFIX = "Describe the Python "
PROMPT_FUNCTION_DESCRIPTION_SUFFIX = (
    " delimited by triple backticks, including the purpose of each parameter, and document " 
    "the mathematical operations or procedures it performs.  For the mathematical operations "
    "generate LaTex code that can be used to display the equations in a markdown document."
)
PROMPT_CLASS_DESCRIPTION_SUFFIX = (
    " delimited by triple backticks, a high-level summary of the class"
)

PROMPT_MODULE_DESCRIPTION_SUFFIX = " delimited by triple backticks with a high-level summary of the module."

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
    completion_prompt = f"{prompt}```{source_code}```"
    # print(f"completion prompt: {completion_prompt}")

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": completion_prompt}],
        max_tokens=8096,
        temperature=0.01,
        n=1,
        stop=None,
    )

    documentation_text = response.choices[0].message.content.strip()
    return documentation_text

def adjust_latex_equations(documentation_text):
    """
    Adjusts the LaTex equations in the documentation text to be compatible with markdown.
    :param documentation_text: generated markdown text with LaTex equations

    :return: documentation_text with LaTex equations adjusted to be compatible with markdown
    """
    # replace \[ with \n$$
    documentation_text = documentation_text.replace("\\[", "\n$$")

    # replace \] with $$
    documentation_text = documentation_text.replace("\\]", "$$")

    return documentation_text


# main function
def main():
    # initialize variables
    total_api_calls = 0
    start_time = datetime.now()

    # Retrieve API from json file
    with open('/openai/.openai/api_key.json') as f:
        api = json.load(f)

    # set API key
    openai.api_key = api['key']

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
            f"{PROMPT_DESCRIPTION_PREFIX} code {PROMPT_MODULE_DESCRIPTION_SUFFIX}",
            source_code
        )
        total_api_calls += 1

        # initialize list of documentation text with module documentation
        documentation_text_list.append(
            f"# Module:`{module_name}` Overview\n\n{module_description}\n\n"
        )
    except openai.error.InvalidRequestError as e:
        print(f"Error generating module level documentation: {e}, bypassing this documentation section")
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
                prompt = f"{PROMPT_DESCRIPTION_PREFIX} function {function_name} {PROMPT_FUNCTION_DESCRIPTION_SUFFIX}"
                documentation_text = generate_documentation(prompt, function_source)
                total_api_calls += 1

                # if generated documentation contains latex equations, need to adjust to be markdown compatible
                documentation_text = adjust_latex_equations(documentation_text)

                # collect generated documentation
                documentation_text_list.append(f"## Function **`{function_name}`** Overview\n{documentation_text}\n\n")
            except openai.error.InvalidRequestError as e:
                print(f"Error generating function overview documentation: {e}, bypassing this documentation section")
                documentation_text_list.append(
                    f"## Function **`{function_name}`** Overview\n"
                    f"### **Error in generating function overview documentation**\n\n"
                )

            # # generate detailed description of function
            # try:
            #     prompt = PROMPT_DETAILS
            #     documentation_text = generate_documentation(prompt, function_source)
            #     total_api_calls += 1
            #     documentation_text_list.append(f"### **Function Details**\n{documentation_text}\n\n")
            # except openai.error.InvalidRequestError as e:
            #     print(f"Error generating function detail documentation: {e}, bypassing this documentation section")
            #     documentation_text_list.append(
            #         f"### **Error in generating function detail documentation**\n\n"
            #     )


        # generate documentation for classes
        elif isinstance(node, ast.ClassDef):
            # extract relevant class information
            class_name = node.name
            class_source = ast.get_source_segment(source_code, node)

            # generate overview of class
            try:
                prompt = f"{PROMPT_DESCRIPTION_PREFIX} class {class_name} {PROMPT_CLASS_DESCRIPTION_SUFFIX}"
                documentation_text = generate_documentation(prompt, class_source)
                total_api_calls += 1
                documentation_text_list.append(f"## Class **`{class_name}`** Overview\n{documentation_text}\n\n")
            except openai.error.InvalidRequestError as e:
                print(f"Error generating class level documentation: {e}, bypassing this documentation section")
                documentation_text_list.append(
                    f"# Class **`{class_name}`** Overview\n\n"
                    f"## **Error in generating class level documentation**\n\n"
                )

            # generate documentation for class methods
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    # extract relevant method information
                    method_name = class_node.name
                    print(f"generating doc for method {node.name}.{method_name}()")
                    method_source = ast.get_source_segment(source_code, class_node)

                    # generate overview of function
                    try:
                        prompt = f"{PROMPT_DESCRIPTION_PREFIX} method {method_name} {PROMPT_FUNCTION_DESCRIPTION_SUFFIX}"
                        documentation_text = generate_documentation(prompt, method_source)
                        total_api_calls += 1

                        # if generated documentation contains latex equations, need to adjust to be markdown compatible
                        documentation_text = adjust_latex_equations(documentation_text)

                        # collect generated documentation
                        documentation_text_list.append(f"### Method **`{method_name}`** Overview\n{documentation_text}\n\n")
                    except openai.error.InvalidRequestError as e:
                        print(f"Error generating method overview documentation: {e}, bypassing this documentation section")
                        documentation_text_list.append(
                            f"## Function **`{method_name}`** Overview\n"
                            f"### **Error in generating method overview documentation**\n\n"
                        )

                    # # generate detailed description of function
                    # try:
                    #     prompt = PROMPT_DETAILS
                    #     documentation_text = generate_documentation(prompt, method_source)
                    #     total_api_calls += 1
                    #     documentation_text_list.append(f"#### **Method Details**\n{documentation_text}\n\n")
                    # except openai.error.InvalidRequestError as e:
                    #     print(f"Error generating method detail documentation: {e}, bypassing this documentation section")
                    #     documentation_text_list.append(
                    #         f"### **Error in generating method detail documentation**\n\n"
                    #     )

    # consolidate all the generated documentation text into single markdown file
    documentation_text = "".join(documentation_text_list)

    # output documentation text
    with open(documentation_file, 'w') as f:
        f.write(documentation_text)

    elapsed_time = datetime.now() - start_time

    print(
        f"finished documenting {args.source_file}...created {documentation_file} "
        f"elapsed time {elapsed_time.total_seconds():01.0f} seconds "
        f"total LLM api calls {total_api_calls}"
    )


if __name__ == '__main__':
    main()






