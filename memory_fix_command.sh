#!/bin/bash

# Ultra memory-optimized training command for single A100 GPU
# This uses the most aggressive memory-saving settings

APPTAINERENV_PYTHONPATH="/app/src:/home/jchen396/gr00t_n1.6" \
apptainer exec --nv \
  --bind /home/jchen396/:/home/jchen396/ \
  --bind /scratch/akriege1/data:/data \
  openpi.sif \
  python3 /app/scripts/train.py pi05_gr00t_local \
    --exp-name dvrk_test_ultra_memory_optimized \
    --data.modality-config-path /home/jchen396/gr00t_n1.6/examples/dVRK/dVRK_config.py \
    --data.dataset-path /data/open_h_suturing \
    --data.embodiment-tag dvrk \
    --optimization_level conservative \
    --hardware_setup single_gpu \
    --enable_gradient_accumulation \
    --gradient_accumulation_steps 8 \
    --overwrite \
    --batch_size 1 \
    --target_memory_utilization 0.5 \
    --jit_warmup_iterations 1 \
    --enable_memory_optimization \
    --enable_mixed_precision \
    --num_workers 2 \
    --fsdp_devices 1