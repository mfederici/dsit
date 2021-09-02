#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --mem=10G
#SBATCH --cpus-per-task=32
#SBATCH --time 48:00:00
#SBATCH --ntasks=1
#SBATCH --priority=TOP
#SBATCH --job-name=sweep
#SBATCH --output=log.txt
#SBATCH --verbose

# Cuda variables
if [ $DEVICE_NAME = das5 ]
then
    echo "Loading modules for das5"
    module load cuda10.1/toolkit

    # Activate conda
    export PATH=/var/scratch/$USER/anaconda3/envs/dl-kit/bin/:$PATH

elif [ $DEVICE_NAME = ivi-cluster ]
then
    echo "Setting the variables path for the ivi-cluster"
    export CUDA_HOME=/usr/local/cuda-10.2
    export PATH=${CUDA_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
    export CUDA_CACHE_PATH="$TMPDIR"/.cuda_cache/
    export CUDA_CACHE_DISABLE=0

    # Activate conda
    export PATH=$HOME/anaconda3/envs/dl-kit/bin/:$PATH
else
    echo "No CUDA variable is set"
fi

# Make sure you are in the project directory
cd $HOME/Documents/dsit

# check wandb and python version
which python
which wandb


# Run agent
for I in 1 2 3 4
do
    # Generate cuda stats to check cuda is found
    nvidia-smi
    echo STARTING AGENT $I
    wandb agent $SWEEP &
    sleep 10
done

wait

echo Done