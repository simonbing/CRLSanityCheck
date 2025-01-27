#! /usr/bin/bash

# Loop over 3 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch ../cluster_gpu.sh python ../../citris_experiment.py --model citrisvae --dataset chambers --task real\
   --batch_size 512 --epochs 250 --seed $SEED --lat_dim 16 --run_name citris_real_d16_full_0 \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads_citris \
   --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done
