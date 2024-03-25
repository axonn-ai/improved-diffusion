#!/bin/bash

userid=$(whoami)

if [ ! -d /mnt/bb/${userid}/my-venv ]; then
	echo "Copying to burst buffer"
	cp /lustre/orion/scratch/ssingh37/csc547/my-venv.tar.gz /mnt/bb/${userid}/
	cd /mnt/bb/${userid}/
	tar -xf my-venv.tar.gz
fi

cp /lustre/orion/scratch/ssingh37/csc547/my-env-2/lib/python3.9/site-packages/torch/lib/librccl.so  /mnt/bb/ssingh37/my-venv/lib/python3.9/site-packages/torch/lib/
