#!/bin/bash

source /nfshomes/aranjan2/adi-venv/bin/activate

module load mpi/mpich-x86_64

export PYTHONPATH="${PYTHONPATH}:/nfshomes/aranjan2/adi-venv/improved-diffusion"

cmd="mpirun -np 2 python image_train.py &> out.txt &" 

echo "${cmd}"

eval "${cmd}"

