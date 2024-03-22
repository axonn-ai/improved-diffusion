#!/bin/bash

#SBATCH -A m4641_g
#SBATCH -C gpu&hbm40g
#SBATCH -t 00:30:00
#SBATCH -q regular
#SBATCH -N 32
#SBATCH --gpus-per-node=4
##SBATCH -c 32

export MPICH_GPU_SUPPORT_ENABLED=0
# module load craype-accel-nvidia80

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$((NNODES * 4))

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500 export CUDA_DEVICE_MAX_CONNECTIONS=1
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

MAIN_DIR="/global/u1/p/prajwal/GodonBell/improved-diffusion"
cd $MAIN_DIR
echo $(pwd)

GLOBAL_BATCH_SIZE=2048
BATCH_SIZE_PER_GPU=$((GLOBAL_BATCH_SIZE / GPUS))
echo "Batch Sizer Per GPU: " $BATCH_SIZE_PER_GPU
NUM_CHANNELS=576

# divided batch size (2048) by num gpus (64) to get 32 batch size
# changed num_channels to 416 to get a model size of roughly 550 mil
# changed learning rate according to paper: (1 / ((416 / 128) ^ 0.5)) * 0.0001 = 0.00005547001
MODEL_FLAGS="--image_size 32 --num_channels ${NUM_CHANNELS} --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 0.00005547001 --batch_size ${BATCH_SIZE_PER_GPU}"

OUTDIR="${SCRATCH}/GodonBellData/improved-diffusion/deepspeed_data/exp-${NUM_CHANNELS}-${GPUS}GPU"
mkdir -p $OUTDIR

OUTFILE="${OUTDIR}/run_${GPUS}_${NUM_CHANNELS}x${GLOBAL_BATCH_SIZE}x${BATCH_SIZE_PER_GPU}.txt"

echo $OUTFILE

cmd="srun -N ${NNODES} -n ${GPUS} python scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --deepspeed_config ./scripts/deepspeed/ds_config_128.json | tee ${OUTFILE}"

echo "${cmd}"

eval "${cmd}"
