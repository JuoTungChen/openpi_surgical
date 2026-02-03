# GPU Optimization Usage Examples

This document provides practical examples and best practices for using the GPU utilization optimization features in OpenPI.

## Basic Examples

### Example 1: Single GPU Training with Balanced Optimization

```bash
# Train on single GPU with balanced optimizations
python scripts/train.py \
    --config pi0_libero \
    --exp_name libero_optimized \
    --batch_size 32 \
    --num_workers 4 \
    --optimization_level balanced \
    --hardware_setup single_gpu
```

**Expected Results:**
- GPU utilization: 80-90%
- Memory utilization: 70-80%
- Training speed: 20-30% improvement over baseline

### Example 2: Multi-GPU Training with Aggressive Optimization

```bash
# Train on 4 GPUs with aggressive optimizations
python scripts/train.py \
    --config pi0_libero \
    --exp_name libero_multi_gpu \
    --batch_size 128 \
    --num_workers 8 \
    --fsdp_devices 4 \
    --optimization_level aggressive \
    --hardware_setup multi_gpu \
    --enable_gradient_accumulation \
    --gradient_accumulation_steps 2
```

**Expected Results:**
- GPU utilization: 85-95%
- Memory utilization: 80-90%
- Training speed: 40-60% improvement over baseline
- Near-linear scaling across GPUs

### Example 3: Conservative Training for Debugging

```bash
# Conservative settings for debugging or unstable systems
python scripts/train.py \
    --config pi0_libero \
    --exp_name libero_debug \
    --batch_size 16 \
    --num_workers 2 \
    --optimization_level conservative \
    --target_memory_utilization 0.7 \
    --log_compilation_timing
```

**Expected Results:**
- Stable training with minimal optimizations
- Detailed performance logging
- Safe memory usage
- Good for identifying bottlenecks

## GR00T Dataset Examples

### Example 4: GR00T Local Dataset with Optimizations

```bash
# Train on local GR00T dataset with full optimizations
python scripts/train.py \
    --config pi05_gr00t_local \
    --exp_name gr00t_optimized \
    --batch_size 64 \
    --num_workers 8 \
    --optimization_level aggressive \
    --gr00t_episode_cache_size 32 \
    --gr00t_video_backend torchcodec
```

**Configuration in training config:**
```python
TrainConfig(
    name="pi05_gr00t_local_optimized",
    model=pi0.Pi0Config(pi05=True),
    data=Gr00tLocalLeRobotDataConfig(
        repo_id="local/gr00t_lerobot",
        dataset_path="/path/to/your/dataset",
        embodiment_tag="dvrk",
        episode_cache_size=32,  # Increased for better performance
        video_backend="torchcodec",
        base_config=DataConfig(
            # Optimized repack transforms
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform({
                        "image": {
                            "base_0_rgb": "observation.images.endoscope_left",
                            "left_wrist_0_rgb": "observation.images.wrist_left",
                            "right_wrist_0_rgb": "observation.images.wrist_right",
                        },
                        "state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                    })
                ]
            ),
        ),
    ),
    # Optimization settings
    num_workers=8,
    optimization_level="aggressive",
    hardware_setup="auto",
    enable_jit_cache_warming=True,
    jit_warmup_iterations=5,
    enable_memory_optimization=True,
    target_memory_utilization=0.9,
)
```

### Example 5: Large Scale GR00T Training

```bash
# Large scale training with maximum optimizations
python scripts/train.py \
    --config pi05_gr00t_local \
    --exp_name gr00t_large_scale \
    --batch_size 256 \
    --num_workers 16 \
    --fsdp_devices 8 \
    --optimization_level aggressive \
    --enable_auto_batch_sizing \
    --max_batch_size 512 \
    --enable_gradient_accumulation \
    --gradient_accumulation_steps 4 \
    --gr00t_episode_cache_size 64
```

## Hardware-Specific Examples

### Example 6: RTX 4090 Single GPU Optimization

```bash
# Optimized for RTX 4090 (24GB VRAM)
python scripts/train.py \
    --config pi0_libero \
    --exp_name rtx4090_optimized \
    --batch_size 48 \
    --num_workers 6 \
    --optimization_level aggressive \
    --target_memory_utilization 0.85 \
    --jit_warmup_iterations 4 \
    --gr00t_episode_cache_size 24
```

### Example 7: H100 Multi-GPU Setup

```bash
# Optimized for 8x H100 (80GB VRAM each)
python scripts/train.py \
    --config pi0_libero \
    --exp_name h100_8gpu \
    --batch_size 512 \
    --num_workers 32 \
    --fsdp_devices 8 \
    --optimization_level aggressive \
    --target_memory_utilization 0.9 \
    --enable_auto_batch_sizing \
    --max_batch_size 1024 \
    --enable_gradient_accumulation \
    --gradient_accumulation_steps 2 \
    --gr00t_episode_cache_size 128
```

### Example 8: Low-End GPU (RTX 3060)

```bash
# Conservative settings for RTX 3060 (12GB VRAM)
python scripts/train.py \
    --config pi0_libero \
    --exp_name rtx3060_conservative \
    --batch_size 16 \
    --num_workers 4 \
    --optimization_level conservative \
    --target_memory_utilization 0.75 \
    --gr00t_episode_cache_size 8 \
    --jit_warmup_iterations 2
```

## Advanced Configuration Examples

### Example 9: Custom Optimization Configuration

```python
# Custom training script with advanced optimization config
import openpi.training.config as _config
from openpi.training.optimization_config import OptimizationConfig

# Create custom optimization config
opt_config = OptimizationConfig()

# Customize for your specific use case
opt_config.gpu_monitoring.target_gpu_utilization = 0.92
opt_config.gpu_monitoring.monitoring_interval = 0.5

opt_config.jax_optimization.enable_auto_batch_sizing = True
opt_config.jax_optimization.gradient_accumulation_steps = 8
opt_config.jax_optimization.jit_warmup_iterations = 10

opt_config.data_loading.num_workers = 16
opt_config.data_loading.prefetch_factor = 12
opt_config.data_loading.episode_cache_size = 64

opt_config.multi_gpu.enable_load_balancing = True
opt_config.multi_gpu.communication_backend = "nccl"

# Use in training
config = _config.get_config("pi0_libero")
# Apply custom optimization settings...
```

### Example 10: Environment-Specific Optimization

```bash
# Set environment variables for optimal performance
export JAX_COMPILATION_CACHE_DIR=/fast_ssd/.cache/jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO  # For debugging multi-GPU issues

# Run training with environment optimizations
python scripts/train.py \
    --config pi0_libero \
    --exp_name env_optimized \
    --optimization_level aggressive \
    --hardware_setup multi_gpu
```

## Performance Monitoring Examples

### Example 11: Detailed Performance Monitoring

```bash
# Enable comprehensive performance monitoring
python scripts/train.py \
    --config pi0_libero \
    --exp_name performance_monitoring \
    --enable_gpu_monitoring \
    --enable_bottleneck_detection \
    --enable_performance_suggestions \
    --log_compilation_timing \
    --gpu_monitoring_interval 0.5 \
    --wandb_enabled
```

**Monitoring Output:**
```
[INFO] GPU utilization: 87.3% (target: 85.0%)
[INFO] Memory utilization: 78.2%
[INFO] Data loading time: 0.045s per batch
[INFO] Forward pass time: 0.123s
[INFO] Backward pass time: 0.089s
[WARNING] Bottleneck detected: Data loading slower than computation
[INFO] Suggestion: Increase num_workers from 4 to 6
```

### Example 12: Wandb Performance Dashboard

The system automatically logs performance metrics to wandb:

```python
# Metrics available in wandb
performance_metrics = {
    # GPU utilization
    "gpu/utilization_current": 0.87,
    "gpu/utilization_average": 0.84,
    "gpu/utilization_peak": 0.95,
    
    # Memory usage
    "memory/allocated_gb": 18.5,
    "memory/reserved_gb": 20.0,
    "memory/utilization": 0.78,
    
    # Timing metrics
    "performance/data_loading_time": 0.045,
    "performance/forward_pass_time": 0.123,
    "performance/backward_pass_time": 0.089,
    "performance/gradient_sync_time": 0.012,
    
    # Optimization stats
    "optimization/jit_cache_hits": 156,
    "optimization/jit_cache_misses": 3,
    "optimization/compilation_time": 2.34,
    
    # Bottleneck detection
    "bottlenecks/data_loading": 0,
    "bottlenecks/computation": 0,
    "bottlenecks/memory": 1,
}
```

## Troubleshooting Examples

### Example 13: Debugging Low GPU Utilization

```bash
# Step 1: Run with detailed monitoring
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_low_gpu \
    --optimization_level conservative \
    --enable_bottleneck_detection \
    --log_compilation_timing \
    --gpu_monitoring_interval 0.5

# Step 2: If data loading is the bottleneck, increase workers
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_low_gpu_fix1 \
    --num_workers 8 \
    --prefetch_factor 6

# Step 3: If still low, enable async optimizations
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_low_gpu_fix2 \
    --num_workers 8 \
    --optimization_level balanced \
    --enable_async_video_decoding
```

### Example 14: Debugging Memory Issues

```bash
# Step 1: Run with memory monitoring
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_memory \
    --target_memory_utilization 0.8 \
    --enable_memory_optimization \
    --log_interval 10

# Step 2: If OOM, reduce batch size and enable auto-sizing
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_memory_fix \
    --batch_size 16 \
    --enable_auto_batch_sizing \
    --min_batch_size 8 \
    --max_batch_size 32

# Step 3: Use gradient accumulation for effective larger batches
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_memory_fix2 \
    --batch_size 16 \
    --enable_gradient_accumulation \
    --gradient_accumulation_steps 4  # Effective batch size: 64
```

### Example 15: Debugging Multi-GPU Issues

```bash
# Step 1: Test single GPU first
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_single_gpu \
    --hardware_setup single_gpu \
    --fsdp_devices 1

# Step 2: Enable multi-GPU with debugging
NCCL_DEBUG=INFO python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_multi_gpu \
    --hardware_setup multi_gpu \
    --fsdp_devices 2 \
    --log_sharding_decisions

# Step 3: If communication errors, try different backend
python scripts/train.py \
    --config pi0_libero \
    --exp_name debug_multi_gpu_gloo \
    --hardware_setup multi_gpu \
    --fsdp_devices 2 \
    --communication_backend gloo
```

## Best Practices Summary

### 1. Start Conservative, Scale Up
```bash
# Always start with conservative settings
python scripts/train.py --config your_config --optimization_level conservative

# Then move to balanced
python scripts/train.py --config your_config --optimization_level balanced

# Finally try aggressive if hardware supports it
python scripts/train.py --config your_config --optimization_level aggressive
```

### 2. Monitor and Adjust
```bash
# Always enable monitoring for the first few runs
python scripts/train.py \
    --config your_config \
    --enable_bottleneck_detection \
    --enable_performance_suggestions \
    --wandb_enabled
```

### 3. Hardware-Specific Tuning
```bash
# Let the system auto-detect and optimize
python scripts/train.py --config your_config --hardware_setup auto

# Or specify your setup explicitly
python scripts/train.py --config your_config --hardware_setup single_gpu  # or multi_gpu
```

### 4. Gradual Optimization
```bash
# Step 1: Basic optimizations
python scripts/train.py --config your_config --optimization_level balanced

# Step 2: Add data loading optimizations
python scripts/train.py --config your_config --num_workers 8 --gr00t_episode_cache_size 32

# Step 3: Add memory optimizations
python scripts/train.py --config your_config --enable_memory_optimization --target_memory_utilization 0.9

# Step 4: Add advanced optimizations
python scripts/train.py --config your_config --enable_auto_batch_sizing --enable_gradient_accumulation
```

### 5. Performance Validation
```bash
# Always validate performance improvements
python scripts/train.py \
    --config your_config \
    --exp_name baseline \
    --disable_all_optimizations

python scripts/train.py \
    --config your_config \
    --exp_name optimized \
    --optimization_level balanced

# Compare results in wandb dashboard
```

These examples provide a comprehensive guide for using the GPU optimization features effectively across different hardware configurations and use cases.