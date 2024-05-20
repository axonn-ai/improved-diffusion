#!/bin/bash

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
DATA_DIR="$IMPROVED_DIFFUSION_DIR/datasets/cifar_test"

echo "$DATA_DIR"

# set env variable to store logs and saved model

export PYTHONPATH="$PYTHONPATH:$IMPROVED_DIFFUSION_DIR"

cd $IMPROVED_DIFFUSION_DIR/scripts

run_cmd="python -m pytorch_fid --device cuda --dims 768 $DATA_DIR $IMPROVED_DIFFUSION_DIR/samples/sample_images"

echo "$run_cmd"
eval "$run_cmd"
set +x
