#! /usr/bin/bash

# Loop over 5 random seeds
for SEED in 23871 725 31040 20988 454
do
  sbatch cluster_gpu.sh python ../run_experiment.py --model contrast_crl --dataset contrast_synth --experiment synth_reprod \
   --batch_size 512 --epochs 200 --seed $SEED --lat_dim 5 --run_name contrast_crl_synth_reprod \
   --output_root /work/bd1083/b382081/projects/CausalRepresentationChambers/results  --metrics SHD,MCC
done
