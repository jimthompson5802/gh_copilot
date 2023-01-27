# Using Codex API

Example code for using Codex API

## Documentation:
* [Codex model](https://beta.openai.com/docs/models/codex)
* [Codex User Guidie](https://beta.openai.com/docs/guides/code/code-completion-limited-beta)
* [Codex API documentation](https://beta.openai.com/docs/api-reference/code-completion)

## Contents
| File                 | Description                                                                                                                                                               |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| codex_api_example.py | Demonstrate invoking the Codex api to generate code from a prompt                                                                                                         |
| create_unit_test.py  | For all source modules in a specified directory, generate unit test and save in a specified output directory. |

## Setup
`Dockerfile` in directory `docker` contains the setup for running the example code.  Run `build_image.sh` to create docker image and run example code in the container.