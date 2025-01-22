#! /usr/bin/bash

# Loop over 3 random seeds
for i in {1..3}
do
  SEED=$RANDOM
  sbatch ../cluster_gpu.sh python ../../contrastive_crl_experiment.py --model contrast_crl --dataset lt_crl_benchmark_v1 --task contrast_crl_real \
   --batch_size 512 --epochs 100 --seed $SEED --lat_dim 5 --optim adam --run_name contrast_crl_real_0 \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
   --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done
