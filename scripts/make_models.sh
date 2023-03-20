#!/usr/bin/env bash

# This script is used to generate the models for the project.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $SCRIPT_DIR/../src/model

python make_concat_nn.py
python make_mean_nn.py
