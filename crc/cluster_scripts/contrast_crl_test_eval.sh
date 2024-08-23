#! /usr/bin/bash

for SEED in 9694 7451 27335 17227 12527
do
  sbatch cluster_gpu.sh python ../evaluate.py --model contrast_crl --dataset lt_camera_v1 --experiment scm_2 \
  --root_dir /work/bd1083/b382081/projects/CausalRepresentationChambers/results \
  --run_name contrast_crl_no_changes_4k --metrics MCC,SHD --seed $SEED
done
