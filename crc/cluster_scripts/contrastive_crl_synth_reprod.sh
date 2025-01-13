#! /usr/bin/bash

# Loop over 5 random seeds
#for i in {1..5}
#do
SEED=26705
sbatch cluster_gpu.sh python ../contrastive_crl_experiment.py --model contrast_crl --dataset contrast_synth --task synth_reprod \
 --batch_size 512 --epochs 100 --seed $SEED --lat_dim 5 --optim adam --run_name synth_reprod_refactor_adam_0 \
 --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
 --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
#done
