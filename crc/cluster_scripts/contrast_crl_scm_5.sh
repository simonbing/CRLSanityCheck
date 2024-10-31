#! /usr/bin/bash

# Loop over 5 random seeds
#for i in {1..5}
for SEED in 6636 2157 23195 61 8112
do
#  SEED=$RANDOM
  sbatch cluster_gpu.sh python ../run_experiment.py --model contrast_crl_linear --dataset lt_camera_v1 --task lt_scm_5 \
   --batch_size 512 --epochs 100 --seed $SEED --image_data --lat_dim 3 --run_name contrast_crl_lin_scm_5_sgd \
   --output_root /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
   --metrics SHD,MCC
done