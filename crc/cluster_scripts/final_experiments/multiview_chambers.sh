#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..3}
do
  SEED=$RANDOM
  sbatch ../cluster_gpu.sh python ../../multiview_experiment.py --model multiview_crl \
  --dataset lt_crl_benchmark_v1 --task contrast_crl_real --exp_name buchholz_1 --train_steps 25000 \
  --batch_size 512 --lr 0.0001 --lat_dim 5 --seed $SEED --run_name multiview_chambers_1 \
  --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
  --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done