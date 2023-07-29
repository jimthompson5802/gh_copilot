#!/bin/bash

docker run -it --rm \
  --name codex \
  --user root:root \
  -e PYTHONPATH=/opt/project \
  --entrypoint /bin/bash \
  -p 6000:6000 \
  -v ${PWD}:/opt/project \
  -v ${HOME}/Desktop:/openai \
  -w /opt/project/ \
  openai_codex:py3.9