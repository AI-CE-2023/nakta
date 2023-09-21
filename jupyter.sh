#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=0


#SBATCH --output ./slurm-esm-gpt-o%05j.log
#SBATCH --error ./slurm-esm-gpt-e%05j.log



source "/user/deepfold/anaconda3/etc/profile.d/conda.sh"
# conda activate

# conda run -n base 


# conda activate


# conda run -n base 
conda activate llama3
jupyter lab --no-browser --port=30003 --ip='0.0.0.0'
date

