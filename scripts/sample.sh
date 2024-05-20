#!/bin/bash
#SBATCH -p batch
#SBATCH -A CSC569
#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH -C nvme

# calculating the number of nodes and GPUs
NNODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=8
GPUS=$(( NNODES * GPUS_PER_NODE )) 

# Make LD_LIBRARY_PATH point to wherever your plugin is installed
# this enables the slingshot-11 plugin for RCCL (crucial for inter-node bw)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/ccs/home/adityatomar/aws-ofi-rccl/build/lib"

module load PrgEnv-cray
module load cray-python
module load amd-mixed/5.6.0 
module load libfabric
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CRAY_MPICH_ROOTDIR}/gtl/lib"
source /ccs/home/adityatomar/ast-venv/bin/activate

export MPICH_GPU_SUPPORT_ENABLED=0

# some RCCL env variables
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

IMPROVED_DIFFUSION_DIR="/ccs/home/adityatomar/improved-diffusion"
DATA_DIR="$IMPROVED_DIFFUSION_DIR/datasets/cifar_train"

echo "$DATA_DIR"

# set env variable to store logs and saved model

export PYTHONPATH="$PYTHONPATH:$IMPROVED_DIFFUSION_DIR"

cd $IMPROVED_DIFFUSION_DIR/scripts

GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / GPUS))

echo "GCDs: $GPUS"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "LOCAL_BATCH_SIZE: $LOCAL_BATCH_SIZE"

export OPENAI_LOGDIR=/ccs/home/adityatomar/improved-diffusion/samples

MODEL_PATH="$IMPROVED_DIFFUSION_DIR/logs/adamw/model001000.pt"

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
SAMPLE_FLAGS="--model_path $MODEL_PATH --batch_size $LOCAL_BATCH_SIZE"

chmod 755 $IMPROVED_DIFFUSION_DIR/scripts/get_rank_from_slurm.sh

SCRIPT="python image_sample.py $SAMPLE_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS" 
run_cmd="srun -N $NNODES -n $GPUS -c7 --gpus-per-task=1 --gpu-bind=closest $IMPROVED_DIFFUSION_DIR/scripts/get_rank_from_slurm.sh $SCRIPT &> $IMPROVED_DIFFUSION_DIR/sample_progress.out"

echo "$run_cmd"
eval "$run_cmd"
set +x
