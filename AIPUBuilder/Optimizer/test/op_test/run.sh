#!/bin/bash -e

export AIPUBUILDER_LOG=2
export PYTHONPATH=../../../../:$PYTHONPATH

python3 ../../tools/optimizer_main.py --cfg ./opt.cfg

