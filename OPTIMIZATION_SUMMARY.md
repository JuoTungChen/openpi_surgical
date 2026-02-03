# GPU Utilization Optimization - Changes Summary

## Problem
Sporadic high batch loading times causing poor GPU utilization:
- Step 0: ~407,616 ms (407 seconds!)
- Step 16: ~145,550 ms (145 seconds)
- Step 17: ~37,949 ms (38 seconds)
- Normal steps: ~5ms

This pattern indicates GPU starvation due to data loading bottlenecks.

## Root Causes Identified

1. **Insufficient worker parallelism**: Default `num_workers=2` 
2. **Tiny episode cache**: Default `episode_cache_size=1` causing constant video re-decoding
3. **No prefetch optimization**: Limited batch pre-loading
4. **Video decoding overhead**: Multiple high-resolution video streams per episode

## Changes Made

### 1. DataLoader Prefetch Factor (`data_loader.py`)

**Location**: `openpi/src/openpi/training/data_loader.py` lines ~418-425

**Change**: Added automatic `prefetch_factor=4` when using multiple workers

```python
prefetch_factor = 2  # Default PyTorch value
if num_workers > 0:
    mp_context = multiprocessing.get_context("spawn")
    # Increase prefetch factor for better GPU utilization
    # Each worker will prefetch this many batches ahead
    prefetch_factor = 4
```

**Impact**: Each worker now pre-loads 4 batches ahead, keeping GPU fed during processing

### 2. Episode Cache Size (`gr00t_lerobot_dataset.py`)

**Location**: `openpi/src/openpi/training/gr00t_lerobot_dataset.py` line ~68

**Change**: Increased default `episode_cache_size` from 1 to 16

```python
# Before
episode_cache_size: int = 1

# After  
episode_cache_size: int = 16  # With detailed comment
```

**Impact**: Each worker caches 16 episodes instead of 1, dramatically reducing video re-decoding

### 3. Config Default Workers (`config.py`)

**Location**: `openpi/src/openpi/training/config.py` line ~891

**Change**: Increased default `num_workers` for `pi05_gr00t_local` config from 2 to 8

```python
TrainConfig(
    name="pi05_gr00t_local",
    num_workers=8,  # Increased from default 2
    ...
)
```

**Impact**: 4x more parallel data loading processes

### 4. Config Episode Cache (`config.py`)

**Location**: `openpi/src/openpi/training/config.py` line ~502

**Change**: Updated default `episode_cache_size` from 1 to 16 with documentation

```python
# Episode cache size per worker. Increase for better performance.
# Good rule of thumb: set to (total_episodes / num_workers)
episode_cache_size: int = 16
```

**Impact**: Better defaults for all GR00T dataset configurations

### 5. Documentation (`performance_tuning.md` - NEW)

**Location**: `openpi/docs/performance_tuning.md`

**Content**: Comprehensive 200+ line guide covering:
- Problem diagnosis
- Root cause analysis
- Optimization strategies
- Recommended configurations for small/medium/large datasets
- Monitoring techniques
- Troubleshooting tips

### 6. Documentation Updates

**Files**: 
- `openpi/docs/groot_training_guide.md` - Added link to performance guide
- `openpi/docs/README.md` - Added performance guide to index

## Expected Performance Improvement

### Before (Default Settings)
```
num_workers=2
episode_cache_size=1
prefetch_factor=2 (implicit)

Results:
- Sporadic 100-400s batch loading times
- Frequent video re-decoding
- Poor GPU utilization (0-30% average)
```

### After (Optimized Settings)
```
num_workers=8
episode_cache_size=16
prefetch_factor=4

Expected Results:
- Consistent 5-20ms batch loading times
- Minimal video re-decoding (95%+ cache hit rate)
- High GPU utilization (80-100%)
- 10-50x faster training throughput
```

## Configuration Recommendations

### Small Dataset (< 50 episodes)
```python
num_workers=4
episode_cache_size=16  # Can cache most/all episodes
```

### Medium Dataset (50-200 episodes)
```python
num_workers=8
episode_cache_size=16  # ~25% of dataset per worker
```

### Large Dataset (> 200 episodes)  
```python
num_workers=8
episode_cache_size=8  # Balance memory vs performance
```

## Memory Impact

With 8 workers and cache_size=16:
- Total cached episodes: 8 × 16 = 128 episodes
- Memory per episode: ~100-500 MB (depends on video resolution/length)
- Total cache memory: 12-64 GB

If memory is limited, reduce `episode_cache_size` proportionally.

## Verification Steps

1. **Check batch timing**: Should see consistent ~5-20ms per batch
2. **Monitor GPU**: `nvidia-smi dmon -s um -d 1` should show 80-100% utilization
3. **Check workers**: `ps aux | grep python` should show 8 worker processes
4. **Verify cache hits**: Add logging in dataset to confirm episode reuse

## Files Changed

1. `openpi/src/openpi/training/data_loader.py` - Added prefetch_factor tuning
2. `openpi/src/openpi/training/gr00t_lerobot_dataset.py` - Increased cache default
3. `openpi/src/openpi/training/config.py` - Updated worker/cache defaults + docs
4. `openpi/docs/performance_tuning.md` - NEW comprehensive guide
5. `openpi/docs/groot_training_guide.md` - Added performance guide link
6. `openpi/docs/README.md` - Added performance guide to index

## Next Steps

1. **Test the changes**: Run training and verify batch timing improvements
2. **Monitor resources**: Ensure memory usage is acceptable
3. **Tune if needed**: Adjust `num_workers` and `episode_cache_size` based on hardware
4. **Profile if issues persist**: Use logging to identify remaining bottlenecks

## Rollback if Needed

If these changes cause issues (OOM, worker crashes), you can override:

```bash
uv run scripts/train.py pi05_gr00t_local \
  --num-workers 2 \
  --data.episode-cache-size 1
```

This reverts to the original conservative settings.
