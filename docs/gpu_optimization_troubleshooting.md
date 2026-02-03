# GPU Optimization Troubleshooting Guide

This guide helps diagnose and resolve common performance issues when using the GPU utilization optimization features in OpenPI.

## Quick Diagnosis

### Performance Issue Checklist

Run this quick diagnostic to identify the most likely cause of performance issues:

```bash
# 1. Check system resources
nvidia-smi
htop
df -h  # Check disk space

# 2. Run with detailed monitoring
python scripts/train.py \
    --config your_config \
    --exp_name diagnostic \
    --optimization_level conservative \
    --enable_bottleneck_detection \
    --enable_performance_suggestions \
    --log_compilation_timing \
    --gpu_monitoring_interval 0.5 \
    --num_train_steps 100  # Short run for diagnosis

# 3. Check logs for warnings and suggestions
grep -E "(WARNING|ERROR|Suggestion)" logs/training.log
```

## Common Issues and Solutions

### 1. Low GPU Utilization (<50%)

**Symptoms:**
- GPU utilization consistently below 50%
- Training slower than expected
- GPU memory usage low despite available memory

**Diagnosis:**
```bash
# Check if data loading is the bottleneck
python scripts/train.py \
    --config your_config \
    --enable_bottleneck_detection \
    --gpu_monitoring_interval 0.5
```

**Solutions:**

#### A. Increase Data Loading Workers
```bash
# Try doubling the number of workers
python scripts/train.py \
    --config your_config \
    --num_workers 8  # Increase from default 2-4
```

#### B. Enable Async Video Decoding (for GR00T datasets)
```bash
python scripts/train.py \
    --config your_config \
    --optimization_level balanced  # Enables async video decoding
```

#### C. Increase Prefetch Factor
```bash
# Add to your training config
prefetch_factor: int = 8  # Increase from default 2-4
```

#### D. Enable Parallel Action Processing
```bash
python scripts/train.py \
    --config your_config \
    --optimization_level aggressive  # Enables parallel processing
```

### 2. High Memory Usage / Out of Memory Errors

**Symptoms:**
- CUDA out of memory errors
- High memory warnings in logs
- Training crashes during batch processing

**Diagnosis:**
```bash
# Monitor memory usage
python scripts/train.py \
    --config your_config \
    --enable_memory_optimization \
    --target_memory_utilization 0.8 \
    --log_interval 10
```

**Solutions:**

#### A. Reduce Batch Size
```bash
python scripts/train.py \
    --config your_config \
    --batch_size 16  # Reduce from larger values
```

#### B. Enable Auto Batch Sizing
```bash
python scripts/train.py \
    --config your_config \
    --enable_auto_batch_sizing \
    --min_batch_size 8 \
    --max_batch_size 64
```

#### C. Use Gradient Accumulation
```bash
# Maintain effective batch size with smaller actual batches
python scripts/train.py \
    --config your_config \
    --batch_size 16 \
    --enable_gradient_accumulation \
    --gradient_accumulation_steps 4  # Effective batch size: 64
```

#### D. Reduce Cache Sizes
```bash
python scripts/train.py \
    --config your_config \
    --gr00t_episode_cache_size 8  # Reduce from default 16
    --jit_max_cache_size 50  # Reduce from default 100
```

### 3. Slow Training Startup

**Symptoms:**
- Long delay before first training step
- Extended "Initializing..." messages
- High CPU usage during startup

**Diagnosis:**
```bash
# Check compilation timing
python scripts/train.py \
    --config your_config \
    --log_compilation_timing \
    --jit_warmup_iterations 1  # Reduce for diagnosis
```

**Solutions:**

#### A. Reduce JIT Warmup Iterations
```bash
python scripts/train.py \
    --config your_config \
    --jit_warmup_iterations 2  # Reduce from default 3-5
```

#### B. Enable Persistent Cache
```bash
python scripts/train.py \
    --config your_config \
    --jit_cache_persistence true
```

#### C. Use Smaller Sample for Warmup
```bash
# Modify training config to use smaller batch for warmup
batch_size: int = 32
jit_warmup_batch_size: int = 8  # If available
```

### 4. Multi-GPU Communication Errors

**Symptoms:**
- NCCL errors in logs
- Training hangs during gradient synchronization
- Inconsistent performance across GPUs

**Diagnosis:**
```bash
# Enable NCCL debugging
NCCL_DEBUG=INFO python scripts/train.py \
    --config your_config \
    --hardware_setup multi_gpu \
    --fsdp_devices 2  # Start with 2 GPUs
```

**Solutions:**

#### A. Check GPU Visibility
```bash
# Ensure all GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi
```

#### B. Try Alternative Communication Backend
```bash
python scripts/train.py \
    --config your_config \
    --communication_backend gloo  # Instead of nccl
```

#### C. Reduce Communication Frequency
```bash
python scripts/train.py \
    --config your_config \
    --optimize_gradient_sync false  # Disable optimization
```

#### D. Start with Fewer GPUs
```bash
# Test with 2 GPUs first
python scripts/train.py \
    --config your_config \
    --fsdp_devices 2
```

### 5. Data Loading Bottlenecks

**Symptoms:**
- High data loading times in logs
- CPU usage spikes during data loading
- GPU idle periods between batches

**Diagnosis:**
```bash
# Monitor data loading performance
python scripts/train.py \
    --config your_config \
    --enable_bottleneck_detection \
    --log_interval 10
```

**Solutions:**

#### A. Optimize Worker Count
```bash
# Try different worker counts
for workers in 4 8 12 16; do
    python scripts/train.py \
        --config your_config \
        --num_workers $workers \
        --exp_name "workers_$workers" \
        --num_train_steps 50
done
```

#### B. Enable Persistent Workers
```bash
# Add to training config
persistent_workers: bool = True
pin_memory: bool = True
```

#### C. Increase Episode Cache Size
```bash
python scripts/train.py \
    --config your_config \
    --gr00t_episode_cache_size 32  # Increase from default
```

#### D. Use Faster Storage
```bash
# Move dataset to SSD if on HDD
# Or use RAM disk for small datasets
sudo mount -t tmpfs -o size=20G tmpfs /tmp/dataset
```

### 6. Video Decoding Issues (GR00T Datasets)

**Symptoms:**
- Slow video loading
- Video decoding errors in logs
- Missing video files warnings

**Diagnosis:**
```bash
# Check video backend configuration
python scripts/train.py \
    --config your_config \
    --gr00t_video_backend torchcodec \
    --log_level DEBUG
```

**Solutions:**

#### A. Optimize Video Backend
```bash
python scripts/train.py \
    --config your_config \
    --gr00t_video_backend torchcodec  # Usually fastest
```

#### B. Check Video File Paths
```bash
# Verify video files exist
ls -la /path/to/dataset/videos/
```

#### C. Disable Video Optimization if Problematic
```bash
# Add to your dataset config
optimize_video_for_dataset: bool = False
```

#### D. Use Alternative Video Views
```bash
# Reduce number of video views if some are missing
gr00t_video_views: ["endoscope_left"]  # Use only available views
```

## Advanced Troubleshooting

### Performance Profiling

#### JAX Profiling
```python
# Add to your training script
import jax.profiler

# Start profiling
jax.profiler.start_trace("/tmp/jax_trace")

# ... run training steps ...

# Stop profiling
jax.profiler.stop_trace()

# View in TensorBoard
# tensorboard --logdir /tmp/jax_trace
```

#### System Resource Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1 > gpu_usage.log &

# Monitor CPU and memory
top -b -d 1 > system_usage.log &

# Monitor I/O
iotop -a > io_usage.log &

# Run training
python scripts/train.py --config your_config

# Stop monitoring
pkill nvidia-smi
pkill top
pkill iotop
```

#### Network Profiling (Multi-GPU)
```bash
# Monitor network usage for multi-node training
iftop -i eth0 > network_usage.log &

# Monitor NCCL operations
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL python scripts/train.py \
    --config your_config 2>&1 | tee nccl_debug.log
```

### Environment Debugging

#### Check JAX Configuration
```python
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
print("JAX compilation cache:", jax.config.jax_compilation_cache_dir)
```

#### Check CUDA Configuration
```bash
nvcc --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"
```

#### Check Dependencies
```bash
pip list | grep -E "(jax|torch|numpy|pandas)"
```

### Configuration Debugging

#### Minimal Configuration Test
```python
# Create minimal config for testing
from openpi.training.optimization_config import OptimizationConfig

config = OptimizationConfig()
config.disable_all_optimizations()

# Test with minimal settings
python scripts/train.py \
    --config debug \
    --disable_all_optimizations \
    --num_train_steps 10 \
    --batch_size 2
```

#### Step-by-Step Optimization Enable
```bash
# Step 1: Baseline (no optimizations)
python scripts/train.py --config your_config --disable_all_optimizations

# Step 2: Enable GPU monitoring only
python scripts/train.py --config your_config --enable_gpu_monitoring

# Step 3: Add JIT cache warming
python scripts/train.py --config your_config --enable_jit_cache_warming

# Step 4: Add memory optimization
python scripts/train.py --config your_config --enable_memory_optimization

# Continue adding optimizations one by one...
```

## Error-Specific Solutions

### AttributeError: '_video_path_pattern'
```bash
# This error indicates missing video metadata
# Solution: Check dataset structure or disable video optimization
python scripts/train.py \
    --config your_config \
    --gr00t_apply_action_transforms false  # Disable if problematic
```

### RuntimeWarning: os.fork() incompatible with JAX
```bash
# Solution: Reduce number of workers or use different multiprocessing method
python scripts/train.py \
    --config your_config \
    --num_workers 0  # Disable multiprocessing
```

### NCCL timeout errors
```bash
# Solution: Increase timeout and check network
export NCCL_TIMEOUT=3600  # 1 hour timeout
export NCCL_IB_DISABLE=1   # Disable InfiniBand if problematic
```

### JAX compilation cache errors
```bash
# Solution: Clear cache and restart
rm -rf ~/.cache/jax
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
```

## Performance Benchmarking

### Baseline Measurement
```bash
# Measure baseline performance
python scripts/train.py \
    --config your_config \
    --exp_name baseline \
    --disable_all_optimizations \
    --num_train_steps 100 \
    --wandb_enabled

# Measure optimized performance
python scripts/train.py \
    --config your_config \
    --exp_name optimized \
    --optimization_level balanced \
    --num_train_steps 100 \
    --wandb_enabled
```

### A/B Testing Different Settings
```bash
# Test different worker counts
for workers in 2 4 8 12; do
    python scripts/train.py \
        --config your_config \
        --exp_name "workers_$workers" \
        --num_workers $workers \
        --num_train_steps 50
done

# Test different optimization levels
for level in conservative balanced aggressive; do
    python scripts/train.py \
        --config your_config \
        --exp_name "opt_$level" \
        --optimization_level $level \
        --num_train_steps 50
done
```

## Getting Help

### Information to Collect

When reporting performance issues, please collect:

1. **System Information:**
```bash
nvidia-smi
lscpu
free -h
df -h
```

2. **Training Configuration:**
```bash
# Your exact command line
python scripts/train.py --config your_config --your_flags

# Or your config file content
```

3. **Performance Logs:**
```bash
# Training logs with performance metrics
grep -E "(GPU|Memory|Performance|WARNING|ERROR)" training.log
```

4. **Wandb Dashboard:**
- GPU utilization graphs
- Memory usage graphs
- Training throughput metrics

### Support Channels

1. Check existing documentation and examples
2. Search for similar issues in project repository
3. Enable detailed logging and analyze bottlenecks
4. Create minimal reproduction case
5. Report issue with collected information

This troubleshooting guide should help resolve most common performance issues with the GPU optimization system.