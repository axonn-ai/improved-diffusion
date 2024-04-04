#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node=4
#SBATCH --time=20
#SBATCH --output={output}

source ~/.bashrc_old

export MPICH_GPU_SUPPORT_ENABLED=0
# module load craype-accel-nvidia80


NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
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
export WORLD_SIZE=$GPUS

IMPROVED_DIFFUSION_DIR="/global/homes/s/ssingh37/improved-diffusion/"
DATA_DIR="$IMPROVED_DIFFUSION_DIR/datasets"

export PYTHONPATH="$PYTHONPATH:$IMPROVED_DIFFUSION_DIR"

cd $IMPROVED_DIFFUSION_DIR/scripts

G_COL={g_col}
G_ROW={g_row}
G_DEPTH={g_depth}
CHANNELS={channels}

MP=$(( G_COL * G_ROW * G_DEPTH ))
G_DATA=$(( GPUS / MP  ))

GLOBAL_BATCH_SIZE={gbs}
LOCAL_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / G_DEPTH / G_DATA ))

IMAGE_SIZE={img_sz}

MODEL_FLAGS="--image_size $IMAGE_SIZE --num_channels $CHANNELS --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size $LOCAL_BATCH_SIZE --G_data $G_DATA --G_inter 1 --G_row $G_ROW --G_col $G_COL --G_depth $G_DEPTH --data_dir $DATA_DIR"

SCRIPT="python -u image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"
run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank_from_slurm.sh $SCRIPT --log_interval 1"

echo "$run_cmd"
eval "$run_cmd"
set +x

