#!/bin/bash

source /nfshomes/aranjan2/adi-venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/nfshomes/aranjan2/adi-venv/improved-diffusion"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

cmd="python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS &> out.txt &" 

echo "${cmd}"

eval "${cmd}"

