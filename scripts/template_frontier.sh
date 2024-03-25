#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC569
#SBATCH -t 00:40:00
#SBATCH -N {nodes}
#SBATCH --output={output}
#SBATCH -C nvme


## calculating the number of nodes and GPUs
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8 ## change as per your machine
GPUS=$(( NNODES * GPUS_PER_NODE )) 

userid=$(whoami)
# These are the two things you need to change as per your setup
# 1. Make LD_LIBRARY_PATH point to wherever your plugin is installed
# this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre/orion/scratch/ssingh37/csc547/aws-ofi-rccl/my-env-2/build/lib"
# 2. Make PYTHONPATH point to your local clone of litgpt
export PYTHONPATH="$PYTHONPATH:/lustre/orion/scratch/$userid/csc547/lit-gpt-dev"

# This blob is setting up my python venv, ignore for conda builds
echo "moving environment to burst buffer"
## load venv onto burst buffer
#srun -N $NNODES --ntasks-per-node=1 prepare_venv.sh
## delete old symbolic link
#rm -rf ~/axonn_venv
## create new symbolic link
#ln -s /mnt/bb/ssingh37/axonn_venv ~/axonn_venv

module load PrgEnv-cray
module load cray-python
module load amd-mixed/5.7.0 #this should match with the rocm version your pytorch uses
module load libfabric
#. /ccs/home/$userid/axonn_venv/bin/activate
. /lustre/orion/scratch/ssingh37/csc547/my-env-2/bin/activate


export MPICH_GPU_SUPPORT_ENABLED=0

## some RCCL env variables
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_MIN_NRINGS=4

# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

export OMP_NUM_THREADS=7 

IMPROVED_DIFFUSION_DIR="/lustre/orion/scratch/ssingh37/csc547/improved-diffusion/"
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
run_cmd="srun -N $NNODES -n $GPUS -c7 --gpus-per-task=1 --gpu-bind=closest ./get_rank_from_slurm.sh $SCRIPT --log_interval 1"

echo "$run_cmd"
eval "$run_cmd"
set +x

