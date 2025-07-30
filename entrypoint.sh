#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-diagfusion}
LOGURU_COLORIZE=0 .venv/bin/python main.py container run
