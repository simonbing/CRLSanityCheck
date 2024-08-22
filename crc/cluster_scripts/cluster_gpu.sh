#! /usr/bin/bash

# Run jobs on GPU node
#SBATCH --partition=gpu

#SBATCH --account=bd1083
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem=0

# Activate environment
sourcce ~/.bashrc
source activate crc

"$@"