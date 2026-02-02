# Quick Reference: GR00T Dataset Adapter

## Installation

```bash
# Install gr00t in the same environment as openpi
pip install -e /path/to/gr00t_n1.6
```

## Basic Usage

```python
from openpi.training.gr00t_lerobot_dataset import (
    Gr00tDatasetSpec,
    Gr00tLeRobotTorchDataset
)

# Create dataset
spec = Gr00tDatasetSpec(
    dataset_path="/path/to/dataset",
    embodiment_tag="dvrk",
    action_horizon=16,
)
dataset = Gr00tLeRobotTorchDataset(spec)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, num_workers=4)
```

## Configuration Options

```python
spec = Gr00tDatasetSpec(
    dataset_path="...",           # Required: path to dataset
    embodiment_tag="...",         # Required: e.g., "dvrk", "aloha"
    action_horizon=16,            # Action chunk length
    episode_indices=np.arange(100), # Optional: episode subset
    language_key="task",          # Optional: prompt field
    video_views=["cam1", "cam2"], # Optional: override views
    episode_cache_size=5,         # Cache size per worker
    video_backend="torchcodec",   # Video decoder
)
```

## Output Format

```python
sample = dataset[0]
# {
#     "observation.images.view_name": [H, W, 3] uint8,
#     "observation.state": [state_dim] float32,
#     "actions": [horizon, action_dim] float32,
#     "prompt": scalar string (if language_key set),
# }
```

## Train/Val Split

```python
import numpy as np

# Split episodes 80/20
all_episodes = np.arange(num_episodes)
np.random.shuffle(all_episodes)
split = int(0.8 * len(all_episodes))

train_spec = Gr00tDatasetSpec(
    dataset_path="...",
    embodiment_tag="...",
    episode_indices=all_episodes[:split],
)

val_spec = Gr00tDatasetSpec(
    dataset_path="...",
    embodiment_tag="...",
    episode_indices=all_episodes[split:],
)
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Failed to import gr00t` | Install: `pip install -e /path/to/gr00t` |
| `No valid (episode, step) pairs` | Episodes too short for action_horizon |
| `modality config must include 'action'` | Add embodiment to GR00T or use fallback |
| Video decoding fails | Check video files exist, try different backend |

## Validation Script

```python
# Test dataset before training
dataset = Gr00tLeRobotTorchDataset(spec)
print(f"Length: {len(dataset)}")

sample = dataset[0]
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f"{k}: {v.shape} {v.dtype}")
```

## See Full Documentation

- [Complete Guide](gr00t_dataset_adapter.md)
- [Code Review](gr00t_dataset_review.md)
