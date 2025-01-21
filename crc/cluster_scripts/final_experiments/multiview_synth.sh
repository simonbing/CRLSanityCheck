#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..2}
#for SEED in 30325 9813 11470
do
SEED=$RANDOM
  sbatch ../cluster_gpu.sh python ../../multiview_experiment.py --model multiview_crl \
  --dataset multiview_synth --task synth_re_mix_2 --train_steps 300000 \
  --batch_size 128 --lr 0.0001 --lat_dim 4 --seed $SEED --run_name multiview_synth_0 \
  --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads \
  --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done