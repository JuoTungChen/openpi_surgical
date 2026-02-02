# Performance Tuning Guide for GR00T Data Loading

This guide explains how to optimize GPU utilization when training OpenPI models with GR00T datasets.

## Problem: Poor GPU Utilization

You may observe sporadic high batch loading times (e.g., 400s, 145s, 38s) mixed with normal times (~5ms). This pattern indicates GPU starvation - the GPU is waiting for data while workers decode videos and process episodes.

## Root Causes

1. **Insufficient parallelism**: Default `num_workers=2` doesn't provide enough parallel data loading
2. **Small episode cache**: Default `episode_cache_size=1` causes constant video re-decoding
3. **Limited prefetching**: Not enough batches pre-loaded to keep GPU busy during processing
4. **Video decoding overhead**: Each episode contains multiple high-resolution video streams that must be decoded

## Optimization Strategy

### 1. Increase Number of Workers

The `num_workers` parameter controls how many parallel processes load and preprocess data.

**Recommendation**: Set to 4-8 workers depending on your CPU cores and I/O bandwidth.

```python
TrainConfig(
    name="pi05_gr00t_local",
    num_workers=8,  # Increased from default 2
    ...
)
```

**Trade-offs**:
- More workers = better GPU utilization but higher CPU/memory usage
- Each worker maintains its own episode cache
- Diminishing returns beyond 8-12 workers on typical systems

### 2. Increase Episode Cache Size

The `episode_cache_size` parameter controls how many decoded episodes each worker keeps in memory.

**Recommendation**: Set to `total_episodes / num_workers` for optimal performance.

Example for 100 episodes with 8 workers:
```python
data=Gr00tLocalLeRobotDataConfig(
    dataset_path="/path/to/dataset",
    embodiment_tag="dvrk",
    episode_cache_size=16,  # 100 episodes / 8 workers ≈ 12-16
    ...
)
```

**Trade-offs**:
- Larger cache = less video re-decoding but higher memory usage per worker
- Each cached episode includes decoded video frames for all views
- Memory usage ≈ `num_workers × episode_cache_size × avg_episode_size`

### 3. Prefetch Factor (Automatic)

The data loader now automatically sets `prefetch_factor=4` when `num_workers > 0`, meaning each worker pre-loads 4 batches ahead.

This is handled automatically in `data_loader.py` and requires no configuration.

### 4. Video Backend Selection

GR00T supports multiple video backends with different performance characteristics:

```python
data=Gr00tLocalLeRobotDataConfig(
    video_backend="torchcodec",  # Default, fastest on GPU systems
    # video_backend="pyav",  # Alternative, more compatible
    ...
)
```

**Recommendations**:
- `torchcodec`: Best performance when GPU acceleration is available
- `pyav`: More stable on some systems, slightly slower

## Recommended Configurations

### Small Dataset (< 50 episodes)

```python
TrainConfig(
    name="pi05_gr00t_local",
    num_workers=4,
    data=Gr00tLocalLeRobotDataConfig(
        dataset_path="/path/to/dataset",
        embodiment_tag="dvrk",
        episode_cache_size=16,  # Cache most/all episodes
        video_backend="torchcodec",
    ),
)
```

### Medium Dataset (50-200 episodes)

```python
TrainConfig(
    name="pi05_gr00t_local",
    num_workers=8,
    data=Gr00tLocalLeRobotDataConfig(
        dataset_path="/path/to/dataset",
        embodiment_tag="dvrk",
        episode_cache_size=16,  # ~25% of dataset per worker
        video_backend="torchcodec",
    ),
)
```

### Large Dataset (> 200 episodes)

```python
TrainConfig(
    name="pi05_gr00t_local",
    num_workers=8,
    data=Gr00tLocalLeRobotDataConfig(
        dataset_path="/path/to/dataset",
        embodiment_tag="dvrk",
        episode_cache_size=8,  # Smaller cache to manage memory
        video_backend="torchcodec",
    ),
)
```

## Monitoring Performance

### Check Batch Loading Times

Add timing code in your training loop:

```python
import time

for step, (obs, actions) in enumerate(data_loader):
    start = time.time()
    # ... training step ...
    load_time = (time.time() - start) * 1000
    if step % 10 == 0:
        print(f"Step {step}: {load_time:.0f}ms")
```

**Good performance**: Consistent ~5-20ms per batch
**Bad performance**: Sporadic spikes to 100ms-400s

### Monitor GPU Utilization

Use `nvidia-smi` to check GPU utilization during training:

```bash
nvidia-smi dmon -s um -d 1
```

**Target**: 80-100% GPU utilization throughout training
**Problem**: Fluctuating 0-100% indicates data loading bottleneck

### Monitor Memory Usage

Check worker memory consumption:

```bash
ps aux | grep python | grep train
```

If workers are using too much memory, reduce `episode_cache_size`.

## Advanced Optimizations

### 1. Pin Memory (Automatic)

PyTorch DataLoader automatically enables `pin_memory=True` when using GPU, which speeds up host-to-device transfers.

### 2. Persistent Workers (Automatic)

Workers are kept alive between epochs (`persistent_workers=True`) to avoid re-initialization overhead.

### 3. Mixed Precision Training

Enable mixed precision in your training config to reduce memory bandwidth:

```python
from openpi.training import mixed_precision

TrainConfig(
    mixed_precision=mixed_precision.BF16Policy(),  # or FP16Policy()
    ...
)
```

### 4. Reduce Video Resolution (If Needed)

If video decoding is still a bottleneck, you can downsample videos during dataset creation (GR00T side).

## Troubleshooting

### Still Seeing Sporadic High Latency?

1. **Check disk I/O**: Use `iotop` to verify your storage isn't the bottleneck
2. **Verify video format**: Ensure videos are in an efficient format (MP4/H.264)
3. **Check CPU usage**: `htop` should show all worker processes active
4. **Reduce batch size**: Smaller batches may improve consistency
5. **Profile workers**: Add logging in `gr00t_lerobot_dataset.py` to identify slow operations

### Out of Memory?

1. Reduce `episode_cache_size`
2. Reduce `num_workers`
3. Use gradient checkpointing in model config
4. Reduce batch size

### Workers Hanging?

This can happen with multiprocessing. Try:
1. Setting `num_workers=0` (single-process mode) to verify the issue
2. Checking for deadlocks in custom transforms
3. Ensuring video files aren't corrupted

## Summary

**Quick wins for better GPU utilization:**

1. ✅ Increase `num_workers` to 8
2. ✅ Increase `episode_cache_size` to 16
3. ✅ Use `torchcodec` video backend
4. ✅ Monitor GPU utilization to verify improvements

These changes are now **defaults** in the `pi05_gr00t_local` config, so you should see immediate improvement!
