#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch cluster_gpu.sh python ../run_experiment.py --model pcl --dataset lt_camera_walks_v1_synth_mix --task lt_pcl_1 \
   --batch_size 512 --epochs 1000 --seed $SEED --lat_dim 5 --run_name pcl_synth_mix_lt_1_300e \
   --output_root /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
   --overwrite_data \
   --metrics MCC
done