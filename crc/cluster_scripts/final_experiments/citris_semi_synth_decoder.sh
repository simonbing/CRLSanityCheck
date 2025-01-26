#! /usr/bin/bash

# Loop over 3 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch ../cluster_gpu.sh python ../../citris_experiment.py --model citrisvae --dataset chambers_semi_synth_decoder --task semi_synth_dec \
   --batch_size 512 --epochs 1000 --seed $SEED --lat_dim 5 --run_name citris_semi_synth_d5_0 \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads_citris \
   --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done
