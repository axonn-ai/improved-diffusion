import os
from comm_model import get_configs_for_unet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run', action="store_true")
parser.add_argument('--gpus', type=int, help="number of GPUs")
parser.add_argument('--batch-size', type=int, help="batch size (in #samples)", default=2048)
parser.add_argument('--time', type=int, help="launch wall clock time (in mins)", default=20)
parser.add_argument('--image-size', type=int, default=32)
parser.add_argument('--cache-all', action='store_true', default=False)
#parser.add_argument('--grad-acc', type=int, help="gradient acc degree", default=1)
parser.add_argument('--model', type=str, choices=["550M", "1B", "2B", "4B", "8B"])
parser.add_argument('--manual', action='store_true', default=False, help="run with manual configuration")
parser.add_argument('--config', type=int, nargs='+', help="if --manual, pass Gr,Gc,Gd as tuple here")

args = parser.parse_args()

#megatron_home = "/pscratch/sd/s/ssingh37/Megatron-LM/"
model=args.model
improved_diffusion_home = "/global/homes/s/ssingh37/improved-diffusion/scripts"

## arch

# 175B
if model == "550M":
    channels = 256
elif model == "1B":
    channels = 416
elif model == "2B":
    channels = 512
elif model == "4B":
    channels = 768
elif model == "8B":
    channels = 1024
else:
    raise NotImplementedError

## gbs and sq
gbs=args.batch_size
img_size=args.image_size
GPUS=args.gpus
topk=5
machine="perlmutter"

if not args.manual:
    top_k_configs = get_configs_for_unet(
        global_batch_size_in_samples=args.batch_size,
        sequence_length=img_size * img_size,
        channels=channels, 
        GPUs=GPUS,
        minimum_degree_of_tensor_parallelism=8,
        model_version="v2",
        topk=topk,
        no_dp=False,
        machine=machine,
        limit=None
    )
    print(top_k_configs)
    ## perf hparams
    ctps=top_k_configs["Gc"]
    rtps=top_k_configs["Gr"]
    dtps=top_k_configs["Gd"]
else:
    config = tuple(args.config)
    rtps = [config[0]]
    ctps = [config[1]] 
    dtps = [config[2]]

GPUS_PER_NODE=4

folder = f"{improved_diffusion_home}/logs/per_comm_model/{model}/"

if not os.path.exists(folder):
    os.makedirs(folder)

with open("template_perlmutter.sh") as f:
    template = f.read()

def sanity_checks(gpu):
    mp = ctp * rtp * dtp
    assert gpu % mp == 0
    dp = gpu // mp
    assert gbs % dp == 0
    bs_per_dp = gbs // dp
    assert bs_per_dp % mbs == 0
    assert mbs % dtp == 0

gpu=GPUS
for ctp,rtp,dtp in zip(ctps, rtps, dtps):
    assert  gpu % GPUS_PER_NODE == 0
    nodes = gpu // GPUS_PER_NODE

    dp = gpu // (rtp*ctp*dtp)
    bs_per_dp = gbs // dp
    mbs = bs_per_dp 
    
    try:
        sanity_checks(gpu)
    except AssertionError:
        continue
    exp_name = f"GPUS_{gpu}_BS_{gbs}_MBS_{mbs}_img_size_{img_size}_{rtp}x{ctp}x{dtp}"
    output_file = os.path.join(folder, f"{exp_name}.out")
    script = template.format(
                nodes=nodes,
                g_col=ctp,
                g_row=rtp,
                g_depth=dtp,
                channels=channels,
                output=output_file,
                gbs=gbs,
                img_sz=img_size
                )
    script_file = os.path.join(folder, f"{exp_name}.sh")
    with open(script_file, "w") as f:
        f.write(script)
    print(f"sbatch -t {args.time} {script_file}")
    if args.run:
        import subprocess
        subprocess.run(["sbatch", "-t" , f"{args.time}", f"{script_file}"])
