# GR00T Dataset Adapter for OpenPI Training

## Overview

The `gr00t_lerobot_dataset.py` module provides a bridge between GR00T's data loading infrastructure and OpenPI's training pipeline. This adapter allows you to train OpenPI models using datasets prepared for GR00T, leveraging GR00T's embodiment-specific modality configurations and LeRobot episode loaders.

## Key Features

- **Reuse GR00T configurations**: Automatically loads embodiment-specific modality configs (video views, state keys, action delta indices) from GR00T
- **Fallback mode**: Can load LeRobot datasets even without GR00T embodiment configs
- **Action chunking**: Properly handles temporal action horizons for action prediction
- **Multi-modal support**: Handles images, proprioceptive state, actions, and language prompts
- **Episode caching**: LRU cache per worker process for efficient episode loading
- **Flexible video backends**: Supports torchcodec, decord, and other backends

## Architecture

### Two Operating Modes

#### 1. **GR00T Modality Config Mode** (Recommended)
When the embodiment tag exists in GR00T's `MODALITY_CONFIGS`:
- Uses GR00T's `LeRobotEpisodeLoader` and `extract_step_data`
- Automatically inherits video views, state keys, action configurations
- Handles temporal stacking and delta indices from modality configs

#### 2. **Fallback Mode**
When no GR00T config exists (e.g., new embodiment):
- Reads parquet files directly
- Decodes videos using GR00T's video utils
- Requires manual specification of `video_views` and `action_horizon`
- Useful for datasets with `info.json` but no embodiment config yet

## Usage

### Basic Example

```python
from openpi.training.gr00t_lerobot_dataset import (
    Gr00tDatasetSpec, 
    Gr00tLeRobotTorchDataset
)
import torch.utils.data as torch_data

# Create dataset specification
spec = Gr00tDatasetSpec(
    dataset_path="/path/to/lerobot_dataset",
    embodiment_tag="dvrk",  # Must exist in gr00t.configs.data.embodiment_configs
    action_horizon=16,
    video_backend="torchcodec",
)

# Create dataset
dataset = Gr00tLeRobotTorchDataset(spec)

# Create PyTorch DataLoader
dataloader = torch_data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

# Iterate
for batch in dataloader:
    images = batch["observation.images.front_cam"]  # [B, H, W, 3]
    state = batch["observation.state"]              # [B, state_dim]
    actions = batch["actions"]                      # [B, horizon, action_dim]
    if "prompt" in batch:
        prompts = batch["prompt"]                   # [B,] numpy string array
```

### Advanced Configuration

```python
import numpy as np

spec = Gr00tDatasetSpec(
    dataset_path="/path/to/dataset",
    embodiment_tag="aloha",
    action_horizon=16,
    
    # Restrict to specific episodes (for train/val split)
    episode_indices=np.arange(0, 100),
    
    # Specify language field to use as prompt
    language_key="task",  # or "sub_task", "annotation.human.action.task_description"
    
    # Override video views (if you don't want all views from modality config)
    video_views=["cam_high", "cam_left_wrist"],
    
    # Cache configuration
    episode_cache_size=5,  # Cache 5 episodes per worker
    
    # Video decoding
    video_backend="torchcodec",
    video_backend_kwargs={"color_conversion_library": "filtergraph"},
)

dataset = Gr00tLeRobotTorchDataset(spec)
```

### Fallback Mode (No GR00T Config)

```python
# For datasets without GR00T embodiment config
spec = Gr00tDatasetSpec(
    dataset_path="/path/to/open_h_dataset",
    embodiment_tag="dvrk",  # Not in GR00T configs - will use fallback
    action_horizon=16,
    video_views=["observation.images.top", "observation.images.wrist"],  # Must specify!
)

dataset = Gr00tLeRobotTorchDataset(spec)
```

## Integration with OpenPI Training

### Step 1: Install GR00T

```bash
# Install gr00t in the same environment as openpi
pip install -e /path/to/gr00t_n1.6
```

### Step 2: Prepare Your Dataset

Your dataset should be in LeRobot format with:
```
dataset_root/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl (optional)
├── data/
│   └── chunk-{episode_chunk}/
│       ├── episode_{episode_index}.parquet
│       └── ...
└── videos/
    └── chunk-{episode_chunk}/
        ├── {video_key}_episode_{episode_index}.mp4
        └── ...
```

### Step 3: Create Custom Dataset Class

Integrate with OpenPI's data loading:

```python
# In your training script or config
from openpi.training.data_loader import create_torch_dataset
from openpi.training.gr00t_lerobot_dataset import (
    Gr00tDatasetSpec,
    Gr00tLeRobotTorchDataset,
)

def create_gr00t_torch_dataset(
    data_config, 
    action_horizon, 
    model_config
):
    """Drop-in replacement for create_torch_dataset."""
    spec = Gr00tDatasetSpec(
        dataset_path=data_config.repo_id,  # Assuming repo_id is local path
        embodiment_tag=data_config.embodiment_tag,
        action_horizon=action_horizon,
        episode_indices=data_config.episode_indices,
        language_key=data_config.language_key,
        video_views=data_config.video_views,
    )
    return Gr00tLeRobotTorchDataset(spec)
```

### Step 4: Update Training Config

Modify your training configuration to use the GR00T dataset:

```python
from openpi.training import config

# Option 1: Patch the create_torch_dataset function
import openpi.training.data_loader as data_loader
data_loader.create_torch_dataset = create_gr00t_torch_dataset

# Option 2: Create custom data loader that uses Gr00tLeRobotTorchDataset
# (See data_loader.py for reference)
```

## Data Format

### Input: GR00T LeRobot Dataset
- **Parquet files**: Episode data with columns like `observation.state`, `action`
- **Video files**: MP4 videos for each camera view
- **Metadata**: `info.json`, `episodes.jsonl`, modality configs

### Output: Per-Timestep Samples
Each `dataset[i]` returns a dictionary with:

```python
{
    # Images (uint8, HWC format)
    "observation.images.front_cam": np.ndarray,      # [H, W, 3]
    "observation.images.wrist_cam": np.ndarray,      # [H, W, 3]
    ...
    
    # State (float32, concatenated across state keys)
    "observation.state": np.ndarray,                 # [state_dim]
    
    # Action chunk (float32, action horizon)
    "actions": np.ndarray,                           # [horizon, action_dim]
    
    # Optional prompt (numpy string scalar)
    "prompt": np.ndarray,                            # scalar string
}
```

### Key Processing Steps

1. **Episode-step mapping**: Global index → (episode_id, step_index)
2. **Action horizon handling**: Only sample valid base steps where `step + horizon <= episode_length`
3. **State extraction**: For multi-temporal state configs, extract the "current" timestep (delta=0)
4. **Image selection**: From temporal stack, select the "current" frame (first in list)
5. **Action concatenation**: Concatenate action keys along the last dimension

## Configuration Reference

### Gr00tDatasetSpec

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | `str` | Required | Path to LeRobot dataset root |
| `embodiment_tag` | `str` | Required | Embodiment identifier (e.g., "dvrk", "aloha") |
| `action_horizon` | `int` | 16 | Action chunk length (used in fallback mode) |
| `episode_indices` | `ndarray\|None` | None | Subset of episodes to use (for splits) |
| `language_key` | `str\|None` | None | Which field to use as prompt text |
| `video_views` | `list[str]\|None` | None | Override video views from config |
| `episode_cache_size` | `int` | 1 | LRU cache size per worker process |
| `video_backend` | `str` | "torchcodec" | Video decoding backend |
| `video_backend_kwargs` | `dict\|None` | None | Additional backend parameters |

## Performance Considerations

### Episode Caching
- Each worker process maintains its own LRU cache
- Cache size of 1 is usually sufficient for sequential access
- Increase cache size if using random sampling patterns

### Multi-Worker Loading
```python
# Good: Multiple workers for data loading
dataloader = torch_data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,  # Parallel episode loading
    prefetch_factor=2,
)
```

### Video Decoding
- `torchcodec`: Fast, GPU-optional, recommended
- `decord`: CPU-based, slower but stable
- Choose based on your hardware and dataset

## Common Issues & Solutions

### Issue 1: "Failed to import gr00t"
**Solution**: Install gr00t in the same environment:
```bash
pip install -e /path/to/gr00t_n1.6
```

### Issue 2: "gr00t modality config must include an 'action' modality"
**Solution**: Your embodiment_tag doesn't exist in GR00T's configs. Either:
- Add the config to GR00T's `embodiment_configs.py`
- Use fallback mode by specifying `video_views` explicitly

### Issue 3: "No valid (episode, step) pairs found"
**Solution**: Episodes might be too short for the action horizon. Check:
- Episode lengths in `episodes.jsonl`
- Your `action_horizon` setting
- That `episode_indices` contains valid episode IDs

### Issue 4: Video decoding errors
**Solution**:
- Check video file paths match the pattern in `info.json`
- Verify video files exist and are not corrupted
- Try a different `video_backend`

### Issue 5: Inconsistent video view names
**Solution**: The adapter tries to handle both:
- `"observation.images.{view}"` format (OpenPI standard)
- `"{view}"` format (raw view names)

If views aren't loading, check the keys in your parquet files:
```python
import pandas as pd
df = pd.read_parquet("data/chunk-0/episode_0.parquet")
print([col for col in df.columns if "image" in col.lower()])
```

## Validation Checklist

Before training, verify:

- [ ] GR00T is installed and importable
- [ ] Dataset path exists and contains `meta/info.json`
- [ ] Embodiment tag exists in GR00T configs OR `video_views` are specified
- [ ] Video files are accessible and decodable
- [ ] Dataset length > 0 (check `len(dataset)`)
- [ ] Sample item has expected keys (test `dataset[0]`)
- [ ] Image shapes match your model expectations
- [ ] Action horizon doesn't exceed episode lengths

## Example Validation Script

```python
from openpi.training.gr00t_lerobot_dataset import (
    Gr00tDatasetSpec,
    Gr00tLeRobotTorchDataset,
)

spec = Gr00tDatasetSpec(
    dataset_path="/path/to/dataset",
    embodiment_tag="your_embodiment",
    action_horizon=16,
)

dataset = Gr00tLeRobotTorchDataset(spec)

print(f"Dataset size: {len(dataset)}")
print(f"Number of episodes: {len(dataset._episode_ids)}")

# Test first sample
sample = dataset[0]
print("\nSample keys:", sample.keys())
print("\nData shapes:")
for k, v in sample.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: {v.shape} {v.dtype}")
    else:
        print(f"  {k}: {type(v)}")

# Test last sample (to verify horizon handling)
sample = dataset[-1]
print("\nLast sample actions shape:", sample["actions"].shape)
```

## Contributing

When extending this adapter:

1. **Maintain compatibility**: Ensure output format matches OpenPI's expectations
2. **Test both modes**: Verify changes work in both GR00T config and fallback modes
3. **Add validation**: Check for common errors and provide helpful messages
4. **Update documentation**: Keep this guide in sync with code changes

## See Also

- [OpenPI Training Guide](../README.md)
- [GR00T Data Loading Documentation](../../gr00t_n1.6/gr00t/data/README.md)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
- [Computing Normalization Stats](norm_stats.md)
