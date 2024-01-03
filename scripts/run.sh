#!/bin/bash

source /nfshomes/aranjan2/adi-venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/nfshomes/aranjan2/adi-venv/improved-diffusion"

cmd="python image_train.py &> out.txt &" 

echo "${cmd}"

eval "${cmd}"

