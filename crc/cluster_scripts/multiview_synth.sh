#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..3}
do
SEED=$RANDOM
  sbatch cluster_gpu.sh python ../apps/train_and_evaluate_method.py --method multiview \
  --dataset multiview_synthetic_chambers_indep --task lt_scm_2 --epochs 500 --val_step 10 \
  --bs 4096 --lr 0.00001 --lat_dim 5 --seed $SEED --run_name multiview_synth_chambers_indep_2\
  --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
  --out_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
  --metrics r2
done