#!/bin/bash

# Use debug mode with minimal model to test if the issue is model size
# This uses dummy model variants that are much smaller

APPTAINERENV_PYTHONPATH="/app/src:/home/jchen396/gr00t_n1.6" \
apptainer exec --nv \
  --bind /home/jchen396/:/home/jchen396/ \
  --bind /scratch/akriege1/data:/data \
  openpi.sif \
  python3 /app/scripts/train.py debug \
    --exp-name dvrk_debug_test \
    --overwrite \
    --batch_size 1 \
    --num_train_steps 10