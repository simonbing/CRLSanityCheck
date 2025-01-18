#! /usr/bin/bash

# Loop over 3 random seeds
for i in {1..3}
do
  SEED=$RANDOM
  sbatch cluster_gpu.sh python ../contrastive_crl_experiment.py --model contrast_crl --dataset contrast_semi_synth_mlp --task lt_scm_2 \
   --batch_size 512 --epochs 100 --seed $SEED --lat_dim 5 --optim adam --run_name contrast_crl_semi_synth_mlp_final_0 \
   --data_root /work/bd1083/b382081/projects/CausalRepresentationChambers/data/contrast_crl_latents \
   --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results
done
