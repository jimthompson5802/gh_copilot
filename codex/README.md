# Using Codex API

Example code for using Codex API

## Documentation:
* [Codex model](https://beta.openai.com/docs/models/codex)
* [Codex User Guide](https://beta.openai.com/docs/guides/code/code-completion-limited-beta)
* [Codex API documentation](https://beta.openai.com/docs/api-reference/code-completion)

## Contents
| File                 | Description                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------|
| codex_api_example.py | Demonstrate invoking the Codex api to generate code from a prompt                                             |
| create_unit_test.py  | For all source modules in a specified directory, generate unit test and save in a specified output directory. |
| ludwig_unit_test_generation_results.md | Results of generating unit tests for Ludwig code using Codex API.                                             |
| translate_sas_to_python.py | Translate SAS code to Python code using Codex API.                                                            |

## Setup
`Dockerfile` in directory `docker` contains the setup for running the example code.  Run `build_image.sh` to create docker image and run example code in the container.  `run_codex.sh` starts a docker container with openai environment.
