# Access OpenAI Codex API
import argparse
import ast
from datetime import datetime
import json
import os

import openai

import tiktoken


# Function to generate code review for a source module
def generate_code_review_feedback(prompt, source_code, model="gpt-3.5-turbo-16k"):
    """
    Generate code review feedback using OpenAI's GPT-3.5 Turbo model.

    Parameters
    :param prompt: The prompt for the code review.
    :param source_code: The source code to be reviewed.
    :param model: The model to use for generating the feedback (default: "gpt-3.5-turbo-16k").

    Returns
    :return: The generated code review feedback.

    Example usage:
        feedback = generate_code_review_feedback("Please review my code", "def my_function():\n    print('Hello, world!')")
    """

    completion_prompt = f"{prompt}```{source_code}```"
    # print(f"completion prompt: {completion_prompt}")

    # compute prompt token size
    encoding = tiktoken.encoding_for_model(model)
    token_size = len(encoding.encode(completion_prompt))
    print(f">>>model {model}, encoded source file token size: {token_size}")


    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": completion_prompt}],
        max_tokens=8096,
        temperature=0.1,
        n=1,
        stop=None,
    )

    code_review_text = response.choices[0].message.content.strip()

    review_text_size = len(encoding.encode(code_review_text))
    print(f">>>model {model}, generated code review token size: {review_text_size}")

    return code_review_text


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

    # retrieve review comment file from second parameter in command line
    parser.add_argument("code_review_file", help="markdown file to contain code review comments")

    # retrieve optional model parameter from command line
    parser.add_argument(
        "--model", 
        help="OpenAI model to use for generating code review documentation", 
        default="gpt-3.5-turbo-16k"
    )

    args = parser.parse_args()

    # read in source file
    with open(args.source_file, 'r') as f:
        source_code = f.read()

    print(f"starting to code review {args.source_file}")

    # create file name for python file
    code_review_file = args.code_review_file

    code_review_text_list = []

    # parse source code into abstract syntax tree to pull out lower level details
    tree = ast.parse(source_code)
    for node in tree.body:

        # generate code review for functions
        if isinstance(node, ast.FunctionDef):
            # extract relevant function information
            function_name = node.name
            print(f"{node}, reviewing function {function_name}")
            function_source = ast.get_source_segment(source_code, node)

            try:
                # generate 
                prompt = (
                    # "I have a Python code snippet that I'd like you to review. Please provide feedback in markdown format on any errors, "
                    # "potential bugs, or areas for improvement:"
                    "Review this python function and provide feedback in markdown format.  "
                    "Identify sections of code that follow good coding practices.  "
                    "Make suggestions to improve the code quality. "
                    "It is not necessary to show how to make the code improvements.:\n"
                )
                code_review_text = generate_code_review_feedback(prompt, function_source)
                total_api_calls += 1

                # collect generated documentation
                code_review_text_list.append(f"## Review for Function **`{function_name}`**\n{code_review_text}\n\n")
            except openai.error.InvalidRequestError as e:
                print(f"Error generating function overview documentation: {e}, bypassing this code review section")
                code_review_text_list.append(
                    f"## Review for Function **`{function_name}`**\n"
                    f"### **Error in generating code review**\n\n"
                )
        else:
            # not interested in this source code node, skip it
            print(f"skipping node {node}")

        # # generate documentation for classes
        # elif isinstance(node, ast.ClassDef):
        #     # extract relevant class information
        #     class_name = node.name
        #     class_source = ast.get_source_segment(source_code, node)

        #     # generate overview of class
        #     try:
        #         prompt = f"{PROMPT_DESCRIPTION_PREFIX} class {class_name} {PROMPT_CLASS_DESCRIPTION_SUFFIX}"
        #         documentation_text = generate_documentation(prompt, class_source)
        #         total_api_calls += 1
        #         documentation_text_list.append(f"## Class **`{class_name}`** Overview\n{documentation_text}\n\n")
        #     except openai.error.InvalidRequestError as e:
        #         print(f"Error generating class level documentation: {e}, bypassing this documentation section")
        #         documentation_text_list.append(
        #             f"# Class **`{class_name}`** Overview\n\n"
        #             f"## **Error in generating class level documentation**\n\n"
        #         )

        #     # generate documentation for class methods
        #     for class_node in node.body:
        #         if isinstance(class_node, ast.FunctionDef):
        #             # extract relevant method information
        #             method_name = class_node.name
        #             print(f"generating doc for method {node.name}.{method_name}()")
        #             method_source = ast.get_source_segment(source_code, class_node)

        #             # generate overview of function
        #             try:
        #                 prompt = f"{PROMPT_DESCRIPTION_PREFIX} method {method_name} {PROMPT_FUNCTION_DESCRIPTION_SUFFIX}"
        #                 documentation_text = generate_documentation(prompt, method_source)
        #                 total_api_calls += 1

        #                 # if generated documentation contains latex equations, need to adjust to be markdown compatible
        #                 documentation_text = adjust_latex_equations(documentation_text)

        #                 # collect generated documentation
        #                 documentation_text_list.append(f"### Method **`{method_name}`** Overview\n{documentation_text}\n\n")
        #             except openai.error.InvalidRequestError as e:
        #                 print(f"Error generating method overview documentation: {e}, bypassing this documentation section")
        #                 documentation_text_list.append(
        #                     f"## Function **`{method_name}`** Overview\n"
        #                     f"### **Error in generating method overview documentation**\n\n"
        #                 )


    # consolidate all the generated documentation text into single markdown file
    code_review_text = "".join(code_review_text_list)

    # output documentation text
    with open(code_review_file, 'w') as f:
        f.write(code_review_text)

    elapsed_time = datetime.now() - start_time

    print(
        f"finished code review {args.source_file}...created {code_review_file} "
        f"elapsed time {elapsed_time.total_seconds():01.0f} seconds "
        f"total LLM api calls {total_api_calls}"
    )


if __name__ == '__main__':
    main()






