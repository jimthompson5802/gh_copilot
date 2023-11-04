#!/bin/bash


python_version=${1:-3.10}

# other valid value is 'plain'
progress=${2:-auto}

docker build --progress ${progress} \
  --build-arg PYTHON_VERSION=${python_version} \
	-t openai_codex:py${python_version} \
	-f ./Dockerfile .



#	--no-cache \
