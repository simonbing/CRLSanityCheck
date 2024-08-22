#! /usr/bin/bash

# Run jobs on GPU node
#SBATCH --partition=gpu

#SBATCH --account=bd1083
#SBATCH -o myfile.out
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem=0

# Activate environment
sourcce ~/.bashrc
source activate crc

# Loop over 5 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch python ../train.py --model contrast_crl --dataset lt_camera_v1 --experiment scm_2 \
   --batch_size 512 --epochs 100 --seed $SEED --lat_dim 5 --run_name contrast_crl_no_changes_4k \
   --output_root /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads
done
