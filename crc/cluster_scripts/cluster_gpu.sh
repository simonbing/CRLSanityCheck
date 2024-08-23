#! /usr/bin/bash

# Run jobs on GPU node
#SBATCH --partition=gpu
#SBATCH --gpus=1

#SBATCH --account=bd1083
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=08:00:00
#SBATCH --nodes=1

# Activate environment
source ~/.bashrc
source activate crc

"$@"