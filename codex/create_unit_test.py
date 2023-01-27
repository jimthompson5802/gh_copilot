#/usr/bin/env python3

import argparse
import json
import os

import openai

# if the script is run as a command line script, run the main function
if __name__ == "__main__":
    # the argument parser is used to parse command line arguments
    parser = argparse.ArgumentParser()

    # add an argument called soource_dir, which is the location housing the source code
    parser.add_argument(
        "--source_dir",
        type=str
    )

    # add an argument called test_dir, which specifies the location to save the unit tests
    parser.add_argument(
        "--test_dir",
        type=str,
    )

    # parse the command line arguments and store them in args
    args = parser.parse_args()
    print(f"args.source_dir: {args.source_dir}, args.test_dir: {args.test_dir}")

    # Retrieve OpenAI API Key from json file
    with open('/openai/.openai/api_key.json') as f:
        api = json.load(f)

    # set API key
    openai.api_key = api['key']

    # retrieve source file names
    source_files = os.listdir(args.source_dir)

    # iterate over source files
    for source_file in source_files:
        # read in source module
        # Source module is assumed to contain a single function
        print(f"processing source_file: {source_file}")
        with open(os.path.join(args.source_dir, source_file), 'r') as f:
            source = f.read()

        print(f">>>>source:\n{source}\n\n")

        # create prompt for code completion to generate unit test
        prompt = source + "\n# Unit test\ndef"

        # get code completion for prompt
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # print generated unit test
        print(f"{response.choices[0].text}")

        # write generated unit test to the specified directory
        with open(os.path.join(args.test_dir, "test_" + source_file), 'w') as f:
            f.write(response.choices[0].text)



