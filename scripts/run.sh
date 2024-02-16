#!/bin/bash

#SBATCH -A m4641
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --gpus-per-node=4
#SBATCH -c 32

export MPICH_GPU_SUPPORT_ENABLED=0
# module load craype-accel-nvidia80

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1


# module load conda
# conda activate base
source /pscratch/sd/a/aranjan/my-venv/bin/activate

module load PrgEnv-gnu/8.5.0
module load pytorch/2.1.0-cu12

export PYTHONPATH="${PYTHONPATH}:/pscratch/sd/a/aranjan/my-venv/improved-diffusion"

# batch size = 128/8 (num gpus) = 16
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16"

cmd="srun -n 8 --cpu-bind=cores python image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"

echo "${cmd}"

eval "${cmd}"

