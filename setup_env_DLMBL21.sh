#!/bin/bash

# checkout the correct branch
git checkout dl_mbl_2021

# create an environment - can take time, should be run at the start of image translation exercise.
conda env create --file=conda_environment.yml
conda init bash
conda activate micro_dl
export PYTHONPATH=$PYTHONPATH:$(pwd)
