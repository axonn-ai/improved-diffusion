# A Guide to Training, Sampling, and Evaluating the Improved Diffusion Model with Different Optimizers

This guide assumes you have read and followed the instructions [here](https://github.com/axonn-ai/improved-diffusion/blob/jorge/README.md)

## Training
1. Under the `improved-diffusion` directory, duplicate, edit, and rename the [train_util.py](https://github.com/axonn-ai/improved-diffusion/blob/jorge/improved_diffusion/train_util.py) file for every optimizer you want to use. For example, to use Jorge, implement the optimizer in a file called `train_util_jorge.py`
   - Creating a new file for every optimizer is necessary since some optimizers (like K-FAC) require non-trivial changes to the code and cannot be switched out with a flag and if-statements
2. Setup [image_train.py](https://github.com/axonn-ai/improved-diffusion/blob/jorge/scripts/image_train.py)
   - Import the `TrainLoop` class with a different alias from every `train_util_<optim>.py` file you created:

     ```python
     from improved_diffusion.train_util_<optim> import TrainLoop as TrainLoop<Optim>
     ```
   - Add if-statements in `main()` for the different optimizers you implemented to be chosen when training the model:

     ```python
     if args.optimizer == "AdamW":
        TrainLoopAdamW(
            model=model,
            ...
        ).run_loop()
     elif args.optimizer == "Jorge":
        TrainLoopJorge(
            model=model,
            ...
        ).run_loop()
     elif ...
     else:
        print("INVALID OPTIMIZER CHOSEN")
     ```
    - You can edit `defaults` in `create_argparser()` to change the default argument values for `TrainLoop`, such as the number of iterations `lr_anneal_steps`
3. (For Frontier) Setup a virtual env and install any necessary libraries from [here](https://github.com/axonn-ai/Megatron-AxoNN/blob/tiny-llama/examples/install_everything_on_frontier.sh) (you will definitely need PyTorch and the RCCL plugin)
4. Setup [train.sh](https://github.com/axonn-ai/improved-diffusion/blob/jorge/scripts/train.sh)
   - Change environment variables and load modules based on the cluster you're training on (the current template is for Frontier)
   - Change hard-coded paths to files and directories to those local to your machine
   - Edit the `OPTIMIZER` variable to whichever optimizer you would like to train with (ex. `OPTIMIZER="Jorge"`). This gets passed as an argument to the `image_train.py` file we setup in step 2
   - Add another elif clause to set the `OPENAI_LOGDIR` environment variable to the path where you want the training logs and checkpoints to be stored (you might have to `mkdir` the directory yourself, or else you might run into path-not-found errors):
   - 
     ```sh
     ...
     elif [ $OPTIMIZER == "Jorge" ]; then
     export OPENAI_LOGDIR=$IMPROVED_DIFFUSION_DIR/logs/jorge
     ...
     ```
5. Run `sbatch train.sh` to begin training. To train with a different optimizer, simply change the `OPTIMIZER` variable in `train.sh`

## Sampling
1. In [image_sample.py](https://github.com/axonn-ai/improved-diffusion/blob/jorge/scripts/image_sample.py), set `num_samples` in `create_argparser()` to the number of samples you want to generate
2. In [sample.sh](https://github.com/axonn-ai/improved-diffusion/blob/jorge/scripts/sample.sh), set the `MODEL_PATH` variable to the path to the model from which you want to sample. This model will likely exist in the `logs/<optim>` directory, depending on what env vars you chose to set in step 4 of Training
     - The repo authors suggest sampling from the EMA models, but depending on the number of iterations you trained your model for, sampling from `model` might be better. More here: https://github.com/openai/improved-diffusion/issues/59
     - Some instructions from Training step 4 might also be useful here
3. Run `sbatch sample.sh` to begin sampling. This will generate a `.npz` file containing NumPy arrays of pixel values of the generated samples
4. To convert the arrays to actual images, run `load_samples.py`. This will be helpful when calculating FID scores in the next section

## Evaluation
1. Read and followed install instructions in the [pytorch-fid](https://github.com/mseitzer/pytorch-fid) repo. (For other kinds of evaluations, check out the [guided-diffusion](//github.com/openai/guided-diffusion/tree/main/evaluations) repo)
2. Setup [evaluate.sh](https://github.com/axonn-ai/improved-diffusion/blob/jorge/scripts/evaluate.sh)
      - Add any desired flags to `run_cmd`
      - set `path/to/dataset1` and `path/to/dataset2` to the test images data path and the sample images data path (order does not matter)
      - If your testing dataset is constant, you can save time by storing a compiled `.npz` version of it, and pass its path as one of the paths. More [here](https://github.com/mseitzer/pytorch-fid/tree/master?tab=readme-ov-file#generating-a-compatible-npz-archive-from-a-dataset)
3. On an interactive node, run `bash evaluate.sh`. After some time, the FID score should be printed out
