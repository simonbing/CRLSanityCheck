#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch cluster_gpu.sh python ../apps/train_and_evaluate_method.py --method multiview \
  --dataset lt_camera_v1 --task lt_scm_2 --epochs 200 --val_step 1 \
  --bs 256 --lr 0.0001 --lat_dim 5 --seed $SEED --run_name multiview_test_0 \
  --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
  --out_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
  --metrics r2
done