#!/bin/bash -e

export AIPUBUILDER_LOG=2
export PYTHONPATH=../../../../:$PYTHONPATH
export AIPUPLUGIN_PATH=./
echo $AIPUPLUGIN_PATH
python3 ../../tools/optimizer_main.py --cfg ./opt.cfg

