#! /usr/bin/bash

# Loop over 5 random seeds
for i in {1..5}
do
  SEED=$RANDOM
  sbatch ../cluster_gpu.sh python ../../multiview_experiment.py --model multiview_crl \
  --dataset chambers_semi_synth_decoder --task chambers_semi_synth_less_sens --exp_name buchholz_1_synth_det --train_steps 28500 \
  --batch_size 512 --lr 0.0001 --lat_dim 5 --seed $SEED --run_name multiview_semi_synth_dec_less_sensors_0 \
  --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/chamber_downloads_citris \
  --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done