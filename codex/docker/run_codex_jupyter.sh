#!/bin/bash
set -xe

docker run -it --rm \
  --name codex_jpynb \
  --shm-size=0.2gb \
  -e PYTHONPATH=/opt/project \
  -p 6006:6006 \
  -p 8888:8888 \
  -v ${PWD}:/opt/project \
  -v ${HOME}/Desktop:/openai \
  --entrypoint jupyter-notebook \
  openai_codex:py3.9 \
  --ip 0.0.0.0 --allow-root --notebook-dir /opt/project --NotebookApp.token='' --no-browser
