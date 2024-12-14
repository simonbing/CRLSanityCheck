#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch cluster_gpu.sh python ../apps/train_and_evaluate_method.py --method contrast_crl \
  --dataset contrast_synthetic --task synth_reprod --encoder fc --epochs 200 --val_step 1 \
  --bs 512 --lr 0.0005 --lat_dim 5 --seed $SEED --run_name contrast_crl_synth_reprod \
  --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
  --output_root /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
  --metrics mcc,shd
done