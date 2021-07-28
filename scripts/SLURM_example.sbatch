#!/bin/bash

# SLURM flags (see https://slurm.schedmd.com/sbatch.html)

# Resource allocation
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=16
#SBATCH --time 10:00:00
#SBATCH --ntasks=1

# Other parameters
#SBATCH --priority=TOP
#SBATCH --job-name=<JOB_NAME>
#SBATCH -D <YOUR_HOME_DIRECTORY>
#SBATCH --output=log.txt
#SBATCH --verbose

# Cuda variables
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CUDA_CACHE_PATH="$TMPDIR"/.cuda_cache/
export CUDA_CACHE_DISABLE=0

# Setting the env_variables

# Weights and biases username and password
export WANDB_USER=<WANDB_USERNAME>
export WANDB_PROJECT=<WANDB_PROJECT>

# Parameters for number of cores for data-loading and device on which the model is placed
export NUM_WORKERS=16
export DEVICE=cuda

# Pointing to the directory in which the dataset is stored
export DATA_ROOT=<DATASET_ROOT> #(e.g. /hddstore/datasets)

# We use temp to save the backups since they are uploaded and deleted afterwards
mkdir /tmp/experiments
export EXPERIMENTS_ROOT=/tmp/experiments

#Generate cuda stats to check cuda is found
nvidia-smi
echo Starting

# Make sure you are in the project directory before trying to run the agent (the one containing the train.py file)
cd <PROJECT_DIRECTORY>

# Use python from the pytorch_and_friends environment
export PATH=$HOME/anaconda3/envs/jade/bin/:$PATH

# Check that python and wandb are accessible
which python
which wandb

# Run the agent
echo Starting agent $WANDB_USER/$WANDB_PROJECT/$SWEEP_ID
wandb agent $SWEEP_ID
wait

# Remove all the files for the model backups since wandb is uploading them anyway
# You can change the experiments directory if you want to keep local versions
rm -r /tmp/experiments
