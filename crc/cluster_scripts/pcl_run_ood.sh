#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch cluster_gpu.sh python ../run_ood.py --estimation_model pcl --dataset lt_camera_walks_v1 --task lt_pcl_1 \
   --batch_size 512 --epochs 100 --seed $SEED --image_data --lat_dim 5 --run_name pcl_ood_lt_1_flatten \
   --output_root /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads
done