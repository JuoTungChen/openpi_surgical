#!/bin/bash

# Test with ultra-minimal dummy model configuration
# This should work even with very limited GPU memory

echo "Creating dummy configuration..."
APPTAINERENV_PYTHONPATH="/app/src:/home/jchen396/gr00t_n1.6" \
apptainer exec --nv \
  --bind /home/jchen396/:/home/jchen396/ \
  --bind /scratch/akriege1/data:/data \
  openpi.sif \
  python3 create_dummy_config.py

echo "Running training with dummy configuration..."
APPTAINERENV_PYTHONPATH="/app/src:/home/jchen396/gr00t_n1.6" \
apptainer exec --nv \
  --bind /home/jchen396/:/home/jchen396/ \
  --bind /scratch/akriege1/data:/data \
  openpi.sif \
  python3 -c "
import sys
sys.path.insert(0, '/app/src')
exec(open('create_dummy_config.py').read())

# Now run training
import subprocess
result = subprocess.run([
    'python3', '/app/scripts/train.py', 'dummy_gr00t_local',
    '--exp-name', 'dvrk_dummy_test',
    '--overwrite',
    '--batch_size', '1',
    '--jit_warmup_iterations', '0',
    '--target_memory_utilization', '0.3',
    '--num_train_steps', '10'
], capture_output=True, text=True)

print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
print('Return code:', result.returncode)
"