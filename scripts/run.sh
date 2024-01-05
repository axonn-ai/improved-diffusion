#!/bin/bash

source /nfshomes/aranjan2/adi-venv/bin/activate

module load mpi/mpich-x86_64

export PYTHONPATH="${PYTHONPATH}:/nfshomes/aranjan2/adi-venv/improved-diffusion"

# divided batch size (128) by num gpus (2) to get 64 batch size
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"

cmd="mpirun -np 2 python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --deepspeed_config ./ds_config.json &> out.txt &" 

echo "${cmd}"

eval "${cmd}"

