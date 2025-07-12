#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-random}
LOGURU_COLORIZE=0 .venv/bin/python run_exp.py
