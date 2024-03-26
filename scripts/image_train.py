"""
Train a diffusion model on images.
"""

import random
import numpy as np
import torch as th

seed=123
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, load_dataset
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

import os
import deepspeed
from torch.optim import AdamW

from mpi4py import MPI


def main():
    parser = create_argparser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # dist_util.setup_dist()
    deepspeed.init_distributed()
    logger.configure()
    logger.log("initialized deepspeed")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("creating data loader...")
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    args.local_rank = int(os.environ.get("LOCAL_RANK"))

    opt = AdamW(list(model.parameters()), lr = args.lr, weight_decay=args.weight_decay)

    # training_data=data
    model_engine, optimizer, __, __ = deepspeed.initialize(
        args=args, model=model, optimizer=opt)

    # data_loader._create_dataloader()

    logger.log("training...")
    TrainLoop(
        opt=optimizer,
        model=model_engine,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="/ccs/home/adityaranjan/scratch/my-venv/improved-diffusion/datasets/cifar_data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=20,
        batch_size=1,  # overriden by run script
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
