# GPU Memory Troubleshooting Guide

## Problem
Training fails with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 16067692680 bytes` (~15GB) during JIT warmup.

## Root Cause Analysis
The error occurs during JIT compilation warmup, which means:
1. The model + batch size combination is too large for available GPU memory
2. JIT compilation creates large intermediate tensors during warmup
3. The default batch size (32) from pi05_gr00t_local config is too large for single GPU

## Immediate Solutions

### 1. Use Ultra Memory-Optimized Command
```bash
# Run this command instead of your original one
bash memory_fix_command.sh
```

Key changes:
- **batch_size: 1** (down from 32) - Most impactful change
- **gradient_accumulation_steps: 8** - Maintains effective batch size of 8
- **target_memory_utilization: 0.5** - Very conservative memory usage
- **jit_warmup_iterations: 1** - Minimal JIT warmup
- **num_workers: 2** - Reduced data loading workers

### 2. Check Memory Requirements First
```bash
# Run this to estimate memory needs before training
APPTAINERENV_PYTHONPATH="/app/src:/home/jchen396/gr00t_n1.6" \
apptainer exec --nv \
  --bind /home/jchen396/:/home/jchen396/ \
  --bind /scratch/akriege1/data:/data \
  openpi.sif \
  python3 check_memory_requirements.py
```

## Progressive Memory Reduction Strategy

If the ultra-optimized command still fails, try these steps in order:

### Step 1: Disable JIT Warmup Completely
Add to your command:
```bash
--jit_warmup_iterations 0
```

### Step 2: Use Even Smaller Effective Batch Size
```bash
--batch_size 1
--gradient_accumulation_steps 16  # Effective batch size = 16
```

### Step 3: Disable Mixed Precision (if needed)
```bash
--enable_mixed_precision false
```

### Step 4: Use Debug Mode
```bash
# Use the debug config which has minimal memory requirements
python3 /app/scripts/train.py debug \
  --exp-name dvrk_debug \
  --overwrite
```

## Memory Optimization Hierarchy

1. **Batch Size** (Most Impact)
   - Default: 32 → Recommended: 1
   - Use gradient accumulation to maintain effective batch size

2. **Mixed Precision** (High Impact)
   - Reduces memory by ~50% for model weights and activations
   - Should be enabled unless causing numerical issues

3. **Gradient Accumulation** (Medium Impact)
   - Allows smaller per-step batch sizes
   - Maintains training dynamics with larger effective batch size

4. **JIT Warmup** (Medium Impact)
   - Reduces initial memory spike during compilation
   - Can be disabled entirely if needed

5. **Target Memory Utilization** (Low-Medium Impact)
   - Controls how aggressively memory is used
   - Lower values provide more safety margin

## Expected Memory Usage

For Pi0.5 model (~3B parameters):
- **FP32**: ~36GB total (model + gradients + optimizer + activations)
- **FP16**: ~20GB total (with mixed precision)
- **Batch size 1**: Reduces activation memory significantly

## Monitoring Commands

### Check GPU Memory Usage
```bash
# Inside container
nvidia-smi

# Watch memory usage during training
watch -n 1 nvidia-smi
```

### Check Available Memory Before Training
```bash
# Get detailed memory info
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

## Alternative Approaches

### 1. Use Smaller Model Variant
If memory issues persist, consider using a smaller model:
```bash
# Use pi0 instead of pi05 (smaller model)
python3 /app/scripts/train.py pi0_gr00t_local  # If this config exists
```

### 2. Use CPU Fallback for Testing
```bash
# Force CPU training for testing (very slow but works)
--hardware_setup cpu_only
```

### 3. Use Model Sharding
```bash
# Enable FSDP sharding even on single GPU
--fsdp_devices 1
--enable_memory_optimization true
```

## Success Indicators

You'll know the memory optimization worked when you see:
1. JIT warmup completes without OOM errors
2. Training starts and progresses normally
3. GPU memory usage stays below 90% of available memory

## Next Steps After Memory Fix

Once training starts successfully:
1. Monitor GPU utilization (should be >80%)
2. Gradually increase batch size if memory allows
3. Adjust gradient accumulation steps accordingly
4. Monitor training loss and convergence

## Common Mistakes to Avoid

1. **Don't increase batch size without checking memory first**
2. **Don't disable gradient accumulation when using batch_size=1**
3. **Don't set target_memory_utilization > 0.8 on single GPU**
4. **Don't skip JIT warmup entirely in production** (only for debugging)