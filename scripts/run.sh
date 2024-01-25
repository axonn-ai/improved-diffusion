#!/bin/bash

#SBATCH -N 1
#SBATCH -n 2
#SBATCH -q debug
#SBATCH --time=00:15:00
#SBATCH -A csc547
#SBATCH --gpus-per-node=2

source /ccs/home/adityaranjan/scratch/my-venv/bin/activate
module load amd/5.6.0
module load libfabric

## these lines enable CUDA aware MPI
module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"

## this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lustre/orion/scratch/adityaranjan/csc547/my-venv/aws-ofi-rccl/build/lib"
# export NCCL_DEBUG=INFO
# export FI_CXI_ATS=0

## this improves cross node bandwidth for some cases
# export NCCL_CROSS_NIC=1

# export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH="${PYTHONPATH}:/ccs/home/adityaranjan/scratch/my-venv/improved-diffusion"

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# divided batch size (128) by number of gpus (2) to get 64
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64"

cmd="srun -n 2 python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS" 

echo "${cmd}"

eval "${cmd}"

