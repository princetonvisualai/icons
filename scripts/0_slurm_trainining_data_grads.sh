#!/bin/bash

#SBATCH --job-name=icons_train_grads
# Set output and error file paths using environment variables
#SBATCH --output=./slurm/train_gradient/output_%A_%a.out
#SBATCH --error=./slurm/train_gradient/error_%A_%a.err
#SBATCH --array=0-199
#SBATCH --gres=gpu:a6000:1  # Specify a single GPU type
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2

source xxxx/anaconda3/etc/profile.d/conda.sh # specify the path to your anaconda3
conda activate icons

# Run the modified script
if [ -z "$1" ]; then
    echo "Please provide the required input argument."
    exit 1
fi
bash ./scripts/0_train_grads/0_get_training_data_grads.sh $1

# Example: 
# sbatch './scripts/0_slurm_train_grads.sh' 500
# note that the 500 is the checkpoint number
