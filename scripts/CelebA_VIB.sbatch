#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=16
#SBATCH --time 10:00:00
#SBATCH --ntasks=1
#SBATCH --priority=TOP
#SBATCH --job-name=frame
#SBATCH --output=log.txt
#SBATCH --verbose

# Cuda variables
if [[ $DEVICE_NAME -eq das5]]
then
    echo "Loading modules for das5"
    module load cuda10.1/toolkit
elif [[ $DEVICE_NAME -eq ivi-cluster]]
then
    echo "Setting the variables path for the ivi-cluster"
    export CUDA_HOME=/usr/local/cuda-10.2
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
    export CUDA_CACHE_PATH="$TMPDIR"/.cuda_cache/
    export CUDA_CACHE_DISABLE=0
else
    echo "No CUDA variable is set"
fi


# Make sure you are in the project directory
cd $HOME/Documents/dl-kit

# Activate conda
export PATH=$HOME/anaconda3/envs/dl-kit/bin/:$PATH

# Set the correct device name
export DEVICE_NAME=ivi-cluster

# check wandb and python version
which python
which wandb

# Generate cuda stats to check cuda is found
nvidia-smi
echo Starting

# Sync the local copy of the dataset
echo copying the celeba dataset to the local storage
rsync -rva $HOME/datasets/celeba /ssdstore/datasets
echo done

# Run the program
python train.py +experiment=CelebA_VIB +trainer.max_epochs=100 params.beta=0.05 run.project=CelebA_examples