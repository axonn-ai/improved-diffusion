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
5. Run `sbatch train.sh` to begin training
