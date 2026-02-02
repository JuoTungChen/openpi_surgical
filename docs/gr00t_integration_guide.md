# GR00T Integration Guide: Training OpenPI with GR00T Configs

This guide explains how to reuse GR00T's data loading infrastructure, training configurations, and action representations when training OpenPI models.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Configuration Mapping](#configuration-mapping)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Integrate GR00T and OpenPI?

**Goal**: Train OpenPI models using GR00T's exact data loading setup to ensure:
- ✅ Identical action representations (RELATIVE/ABSOLUTE/HYBRID_RELATIVE)
- ✅ Same action horizons and delta_indices
- ✅ Consistent normalization statistics
- ✅ Matched video views and data sampling
- ✅ Fair model comparison between frameworks

**What This Enables**:
- Use GR00T's YAML configs and modality configs without code duplication
- Load GR00T's local LeRobot datasets with proper embodiment settings
- Apply GR00T's per-dataset normalization statistics
- Train OpenPI models (π0, π0.5, π0-FAST) on GR00T-prepared data

**What Gets Reused from GR00T**:
1. **Modality Configs**: Action representations, delta_indices, video views
2. **Dataset Configs**: Dataset paths, mix ratios, embodiment tags
3. **Data Loader**: `Gr00tLeRobotTorchDataset` adapter
4. **Normalization Stats**: Per-dataset or shared statistics
5. **Training Hyperparameters**: Learning rate, warmup, batch size

---

## Quick Start

### 1. Basic Usage

```python
from openpi.training.gr00t_config_loader import load_gr00t_config
from openpi.models.pi0 import Pi0Config
import openpi.training.config as _config

# Load GR00T config
bundle = load_gr00t_config(
    yaml_path="/path/to/gr00t_n1.6/examples/dVRK/dVRK_multi_config.yaml",
    modality_config_path="/path/to/gr00t_n1.6/examples/dVRK/dVRK_config.py"
)

# Get embodiment info
embodiment = "dvrk"
action_horizon = bundle.get_action_horizon(embodiment)  # e.g., 16
action_dim = bundle.get_action_dim(embodiment)          # e.g., 14

# Create OpenPI model config
model_config = Pi0Config(
    action_horizon=action_horizon,
    action_dim=action_dim,
    max_token_len=512,
)

# Convert to OpenPI training config
openpi_config = bundle.to_openpi_config(
    model_config=model_config,
    exp_name="openpi_dvr_experiment",
    project_name="openpi",
)

# Train!
from openpi.scripts.train import run_training
run_training(openpi_config)
```

### 2. Command-Line Training Script

Create `train_with_groot_config.py`:

```python
#!/usr/bin/env python
"""Train OpenPI model using GR00T configuration."""

import tyro
from openpi.training.gr00t_config_loader import load_gr00t_config
from openpi.models.pi0 import Pi0Config
from openpi.scripts.train import run_training


def main(
    groot_yaml: str = "examples/dVRK/dVRK_multi_config.yaml",
    groot_modality: str = "examples/dVRK/dVRK_config.py",
    embodiment: str = "dvrk",
    exp_name: str = "openpi_from_groot",
    max_token_len: int = 512,
):
    """Train OpenPI using GR00T configs.
    
    Args:
        groot_yaml: Path to GR00T YAML config
        groot_modality: Path to GR00T modality config (Python file)
        embodiment: Embodiment tag to train on
        exp_name: Experiment name for checkpoints
        max_token_len: Max token length for OpenPI model
    """
    # Load GR00T config
    bundle = load_gr00t_config(groot_yaml, groot_modality)
    
    # Create model config with GR00T's action settings
    model_config = Pi0Config(
        action_horizon=bundle.get_action_horizon(embodiment),
        action_dim=bundle.get_action_dim(embodiment),
        max_token_len=max_token_len,
    )
    
    # Convert to OpenPI config
    openpi_config = bundle.to_openpi_config(
        model_config=model_config,
        exp_name=exp_name,
    )
    
    print(f"Training OpenPI on embodiment: {embodiment}")
    print(f"  Action horizon: {model_config.action_horizon}")
    print(f"  Action dim: {model_config.action_dim}")
    print(f"  Dataset: {bundle.dataset_configs[0].dataset_paths[0]}")
    
    # Run training
    run_training(openpi_config)


if __name__ == "__main__":
    tyro.cli(main)
```

Run with:
```bash
python train_with_groot_config.py \
  --groot-yaml /path/to/gr00t_n1.6/examples/dVRK/dVRK_multi_config.yaml \
  --groot-modality /path/to/gr00t_n1.6/examples/dVRK/dVRK_config.py \
  --embodiment dvrk \
  --exp-name my_openpi_experiment
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GR00T Repo                               │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Modality Configs (embodiment_configs.py)             │     │
│  │  - Action representations (RELATIVE/ABSOLUTE)         │     │
│  │  - Delta indices (action horizon)                     │     │
│  │  - Video views, state keys                            │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  YAML Configs (examples/*/config.yaml)                │     │
│  │  - Dataset paths and mix ratios                       │     │
│  │  - Training hyperparameters                           │     │
│  │  - Normalization stats paths                          │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Data Loaders                                          │     │
│  │  - LeRobotEpisodeLoader                               │     │
│  │  - ShardedSingleStepDataset                           │     │
│  │  - StateActionProcessor                               │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Import & Load
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OpenPI Integration Layer                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  gr00t_config_loader.py                                │     │
│  │  - load_gr00t_config()                                │     │
│  │  - Gr00tConfigBundle                                   │     │
│  │  - to_openpi_config()                                 │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  gr00t_lerobot_dataset.py                             │     │
│  │  - Gr00tLeRobotTorchDataset                           │     │
│  │  - Adapts GR00T loader to OpenPI format              │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Train
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OpenPI Training                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Models (π0, π0.5, π0-FAST)                           │     │
│  │  - JAX/Flax implementation                             │     │
│  │  - Vision encoder + LLM + action head                 │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Training Loop (scripts/train.py)                      │     │
│  │  - Data loading, forward pass, loss, optimization     │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
GR00T YAML Config
    │
    ├─► Dataset paths
    ├─► Embodiment tags
    ├─► Mix ratios
    └─► Training params
        │
        ▼
Gr00tConfigBundle
    │
    ├─► Extract action_horizon
    ├─► Extract action_dim
    ├─► Extract video_views
    └─► Create OpenPI config
        │
        ▼
OpenPI TrainConfig
    │
    └─► Gr00tLocalLeRobotDataConfig
            │
            ▼
        Gr00tLeRobotTorchDataset (adapter)
            │
            ├─► Uses GR00T modality configs
            ├─► Loads from local LeRobot dataset
            └─► Returns samples in OpenPI format:
                {
                  "observation.images.front": [H,W,3],
                  "observation.images.wrist": [H,W,3],
                  "observation.state": [state_dim],
                  "actions": [horizon, action_dim],
                  "prompt": str
                }
                    │
                    ▼
                OpenPI transforms (resize, normalize, tokenize)
                    │
                    ▼
                Model forward pass
```

---

## Step-by-Step Setup

### Step 1: Install GR00T

First, ensure GR00T is installed in your environment:

```bash
# Option 1: Install as editable package
pip install -e /path/to/gr00t_n1.6

# Option 2: Add to PYTHONPATH
export PYTHONPATH="/path/to/gr00t_n1.6:$PYTHONPATH"
```

Verify installation:
```python
import gr00t
from gr00t.configs.base_config import get_default_config
print("GR00T installed successfully!")
```

### Step 2: Locate GR00T Config Files

For your embodiment, you need:

1. **YAML Config** (e.g., `examples/dVRK/dVRK_multi_config.yaml`):
   ```yaml
   data:
     datasets:
       - dataset_paths:
           - /path/to/dvr_dataset
         embodiment_tag: dvrk
         mix_ratio: 1.0
     percentile_stats_path: /path/to/stats.json
     video_backend: torchcodec
   
   training:
     max_steps: 50000
     global_batch_size: 256
     learning_rate: 1e-4
     warmup_steps: 1000
   
   model:
     # ... model config ...
   ```

2. **Modality Config** (e.g., `examples/dVRK/dVRK_config.py`):
   ```python
   from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
   from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation
   
   MODALITY_CONFIGS["dvrk"] = {
       "action": ModalityConfig(
           delta_indices=list(range(16)),  # 16-step horizon
           modality_keys=["left_eef", "left_gripper", "right_eef", "right_gripper"],
           action_configs=[
               ActionConfig(
                   rep=ActionRepresentation.RELATIVE,
                   # ... other config ...
               ),
               # ... more action configs ...
           ],
       ),
       "observation": ModalityConfig(
           delta_indices=[0],
           modality_keys=["front_cam", "wrist_cam"],
       ),
   }
   ```

### Step 3: Load GR00T Config in OpenPI

```python
from openpi.training.gr00t_config_loader import load_gr00t_config

bundle = load_gr00t_config(
    yaml_path="/path/to/gr00t_n1.6/examples/dVRK/dVRK_multi_config.yaml",
    modality_config_path="/path/to/gr00t_n1.6/examples/dVRK/dVRK_config.py",
    load_statistics=True,  # Load normalization stats if available
)

# Inspect what was loaded
print(f"Datasets: {len(bundle.dataset_configs)}")
for ds in bundle.dataset_configs:
    print(f"  - {ds.embodiment_tag}: {ds.dataset_paths[0]}")

print(f"\nEmbodiments: {list(bundle.modality_configs.keys())}")
```

### Step 4: Create OpenPI Model Config

```python
from openpi.models.pi0 import Pi0Config

embodiment = "dvrk"

model_config = Pi0Config(
    # Get these from GR00T config
    action_horizon=bundle.get_action_horizon(embodiment),
    action_dim=bundle.get_action_dim(embodiment),
    
    # OpenPI-specific settings
    max_token_len=512,
    model_type="pi0_5",  # or "pi0", "pi0_fast"
    
    # Optional: customize architecture
    # num_transformer_layers=8,
    # hidden_size=768,
)

print(f"Model config:")
print(f"  Action horizon: {model_config.action_horizon}")
print(f"  Action dim: {model_config.action_dim}")
```

### Step 5: Convert to OpenPI TrainConfig

```python
openpi_config = bundle.to_openpi_config(
    model_config=model_config,
    exp_name="openpi_dvrk_experiment",
    project_name="openpi",
    checkpoint_base_dir="./checkpoints",
    assets_base_dir="./assets",
)

# The resulting config will:
# - Use Gr00tLocalLeRobotDataConfig for data loading
# - Load from GR00T's dataset path
# - Use GR00T's embodiment tag and modality config
# - Apply GR00T's training hyperparameters
```

### Step 6: Train!

```python
# Option A: Use OpenPI's training function
from openpi.scripts.train import run_training
run_training(openpi_config)

# Option B: Use lower-level training loop
from openpi.scripts.train import init_train_state, train_step
import jax

rng = jax.random.key(42)
mesh = create_mesh(...)  # Set up device mesh
train_state, sharding = init_train_state(openpi_config, rng, mesh, resume=False)

# Training loop
for step in range(openpi_config.num_train_steps):
    batch = next(data_loader)
    rng, step_rng = jax.random.split(rng)
    train_state, metrics = train_step(openpi_config, step_rng, train_state, batch)
    # Log metrics, checkpoint, etc.
```

---

## Configuration Mapping

### GR00T → OpenPI Config Fields

| GR00T Config | OpenPI Config | Notes |
|-------------|---------------|-------|
| `data.datasets[0].dataset_paths[0]` | `data.gr00t_dataset_path` | Dataset directory path |
| `data.datasets[0].embodiment_tag` | `data.gr00t_embodiment_tag` | Embodiment identifier |
| `modality_configs[tag]["action"].delta_indices` | `model.action_horizon` | Length of delta_indices |
| Sum of action_configs dimensions | `model.action_dim` | Total action dimension |
| `modality_configs[tag]["observation"].modality_keys` | `data.gr00t_video_views` | Video view names |
| `training.learning_rate` | `lr_schedule.peak_lr` | Peak learning rate |
| `training.warmup_steps` | `lr_schedule.warmup_steps` | LR warmup steps |
| `training.max_steps` | `num_train_steps` | Total training steps |
| `training.global_batch_size` | `batch_size` | Batch size |
| `training.weight_decay` | `optimizer.weight_decay` | Weight decay |
| `data.video_backend` | `data.gr00t_video_backend` | Video decoder |
| `data.percentile_stats_path` | Statistics loaded via bundle | Norm stats |

### Action Representation Mapping

GR00T's action representations are preserved through the dataset adapter:

| GR00T ActionRepresentation | Meaning | Preserved in OpenPI? |
|---------------------------|---------|----------------------|
| `RELATIVE` | Deltas from current state | ✅ Yes (via Gr00tLeRobotTorchDataset) |
| `ABSOLUTE` | Target positions | ✅ Yes |
| `HYBRID_RELATIVE` | Relative translation + absolute rotation | ✅ Yes |
| `DELTA` | Incremental changes | ✅ Yes |

The `Gr00tLeRobotTorchDataset` adapter internally uses GR00T's `StateActionProcessor` to handle all representation conversions and normalizations before passing data to OpenPI.

---

## Advanced Usage

### Multi-Dataset Training

Currently, OpenPI's config system doesn't have built-in dataset mixing like GR00T. For multi-dataset training:

**Option 1: Sequential Training**
```python
for dataset_config in bundle.dataset_configs:
    embodiment = dataset_config.embodiment_tag
    
    # Create separate config for each dataset
    model_config = Pi0Config(
        action_horizon=bundle.get_action_horizon(embodiment),
        action_dim=bundle.get_action_dim(embodiment),
    )
    
    # Train on this dataset
    openpi_config = bundle.to_openpi_config(...)
    run_training(openpi_config)
```

**Option 2: Manual Dataset Mixing** (requires custom data loader)
```python
# Create multiple datasets
datasets = []
for ds_config in bundle.dataset_configs:
    dataset = Gr00tLeRobotTorchDataset(
        spec=Gr00tDatasetSpec(
            dataset_path=ds_config.dataset_paths[0],
            embodiment_tag=ds_config.embodiment_tag,
            # ...
        )
    )
    datasets.append((dataset, ds_config.mix_ratio))

# Implement weighted sampling
def sample_mixed_batch(datasets, batch_size):
    # Choose dataset according to mix_ratio
    weights = [ratio for _, ratio in datasets]
    dataset_idx = np.random.choice(len(datasets), p=weights / np.sum(weights))
    dataset, _ = datasets[dataset_idx]
    
    # Sample batch from chosen dataset
    indices = np.random.choice(len(dataset), batch_size)
    return [dataset[i] for i in indices]
```

### Custom Repack Transforms

If your dataset's video views don't match OpenPI's expected image keys:

```python
# Get video views from GR00T config
video_views = bundle.get_video_views("dvrk")  # e.g., ["front_cam", "wrist_cam"]

# Create custom mapping
repack_transforms = bundle.create_repack_transforms(
    embodiment_tag="dvrk",
    target_image_keys=["image_primary", "image_wrist"],  # OpenPI format
)

# Add to data config
data_config = bundle.to_openpi_config(...)
data_config = dataclasses.replace(
    data_config,
    data=dataclasses.replace(
        data_config.data,
        repack_transforms=repack_transforms,
    )
)
```

### Using GR00T Statistics in OpenPI

GR00T's normalization statistics can be loaded and used:

```python
bundle = load_gr00t_config(..., load_statistics=True)

if bundle.statistics is not None:
    # Save to OpenPI assets directory
    stats_path = f"./assets/{embodiment}/norm_stats.json"
    bundle.save_statistics(stats_path)
    
    # Or convert to OpenPI NormStats format
    from openpi.transforms import NormStats
    
    # Extract per-key statistics
    openpi_stats = {}
    for key, stats_dict in bundle.statistics.items():
        if "q02" in stats_dict and "q98" in stats_dict:
            # Percentile normalization
            openpi_stats[key] = NormStats(
                min=stats_dict["q02"],
                max=stats_dict["q98"],
                # ...
            )
```

### Debugging Data Loading

To verify that GR00T data is loaded correctly:

```python
from openpi.training.data_loader import create_torch_dataset

# Create dataset
dataset = create_torch_dataset(
    data_config=openpi_config.data,
    action_horizon=openpi_config.model.action_horizon,
    model_config=openpi_config.model,
)

# Sample and inspect
sample = dataset[0]
print("Sample keys:", sample.keys())
print("Image shapes:", {k: v.shape for k, v in sample.items() if "image" in k})
print("Action shape:", sample["actions"].shape)
print("Prompt:", sample.get("prompt", "N/A"))

# Verify action horizon matches
assert sample["actions"].shape[0] == openpi_config.model.action_horizon
```

---

## Troubleshooting

### Common Issues

#### 1. **ImportError: Failed to import gr00t**

**Cause**: GR00T not installed or not in PYTHONPATH

**Solution**:
```bash
pip install -e /path/to/gr00t_n1.6
# or
export PYTHONPATH="/path/to/gr00t_n1.6:$PYTHONPATH"
```

#### 2. **KeyError: Embodiment 'X' not found in modality configs**

**Cause**: Modality config Python file not loaded, or typo in embodiment tag

**Solution**:
- Verify modality config path is correct
- Check that the Python file registers the embodiment in `MODALITY_CONFIGS`
- Ensure embodiment tag matches exactly (case-sensitive)

#### 3. **Action dimension mismatch**

**Cause**: `get_action_dim()` heuristic doesn't match actual dimension

**Solution**: Manually specify action_dim:
```python
model_config = Pi0Config(
    action_horizon=bundle.get_action_horizon("dvrk"),
    action_dim=14,  # Specify manually instead of get_action_dim()
)
```

To find correct dimension, check GR00T's modality config:
```python
modality_config = bundle.modality_configs["dvrk"]["action"]
for key, cfg in zip(modality_config.modality_keys, modality_config.action_configs):
    print(f"{key}: format={cfg.format}, type={cfg.type}")
```

#### 4. **Video views not loading**

**Cause**: Video view names in GR00T config don't match dataset

**Solution**: Explicitly specify video views:
```python
openpi_config = bundle.to_openpi_config(...)
openpi_config = dataclasses.replace(
    openpi_config,
    data=dataclasses.replace(
        openpi_config.data,
        gr00t_video_views=["camera_0", "camera_1"],  # Actual view names in dataset
    )
)
```

#### 5. **Normalization statistics not found**

**Cause**: `percentile_stats_path` in YAML is None or file doesn't exist

**Solution**:
- Run GR00T's statistics computation: `scripts/compute_norm_stats.py`
- Or disable statistics: `load_statistics=False`
- Or compute OpenPI statistics separately

#### 6. **Multi-dataset mixing not working**

**Cause**: OpenPI doesn't have built-in dataset mixing

**Solution**: See [Multi-Dataset Training](#multi-dataset-training) section above

---

## Comparison: Training with GR00T Config vs Native OpenPI

### With GR00T Config (Recommended for Consistency)

```python
from openpi.training.gr00t_config_loader import load_gr00t_config

# Load GR00T config
bundle = load_gr00t_config(yaml_path="...", modality_config_path="...")

# Create model config using GR00T settings
model_config = Pi0Config(
    action_horizon=bundle.get_action_horizon("dvrk"),
    action_dim=bundle.get_action_dim("dvrk"),
)

# Train
openpi_config = bundle.to_openpi_config(model_config=model_config, ...)
run_training(openpi_config)
```

**Advantages**:
- ✅ Guaranteed consistency with GR00T training setup
- ✅ Reuses GR00T's action representations and configs
- ✅ Easy to compare model performance between frameworks
- ✅ Single source of truth for embodiment settings

**Disadvantages**:
- ❌ Requires GR00T installation
- ❌ Two config systems to understand
- ❌ Limited multi-dataset mixing support

### Native OpenPI Config

```python
from openpi.training.config import LeRobotDataConfig
from openpi.models.pi0 import Pi0Config

# Manually configure everything
data_config = LeRobotDataConfig(
    repo_id="physicalintelligence/dvrk_dataset",
    assets=AssetsConfig(...),
    # ... manually set all options ...
)

model_config = Pi0Config(
    action_horizon=16,  # Manually set
    action_dim=14,      # Manually set
)

train_config = TrainConfig(
    data=data_config,
    model=model_config,
    # ... set all training params ...
)

run_training(train_config)
```

**Advantages**:
- ✅ No GR00T dependency
- ✅ Full OpenPI feature set
- ✅ Simpler for OpenPI-only users

**Disadvantages**:
- ❌ Risk of misconfiguration vs GR00T
- ❌ Harder to ensure parity with GR00T training
- ❌ Must manually compute/copy normalization stats

---

## Summary

### Key Takeaways

1. **Use `load_gr00t_config()`** to import GR00T YAML and modality configs
2. **Extract settings** using `get_action_horizon()`, `get_action_dim()`, etc.
3. **Convert to OpenPI** using `to_openpi_config()`
4. **Data flows through** `Gr00tLeRobotTorchDataset` adapter
5. **Action representations preserved** via GR00T's StateActionProcessor

### When to Use This Integration

✅ **Use when**:
- Training OpenPI on GR00T-prepared datasets
- Comparing OpenPI vs GR00T models fairly
- You have existing GR00T configs you want to reuse
- Need exact same action representations and horizons

❌ **Don't use when**:
- Training on HuggingFace LeRobot datasets (use native OpenPI)
- Don't need GR00T compatibility
- GR00T installation not feasible

### Next Steps

1. **Set up environment**: Install GR00T and OpenPI
2. **Locate configs**: Find your embodiment's YAML and modality config
3. **Create training script**: Use template from [Quick Start](#quick-start)
4. **Run training**: Execute with your dataset
5. **Monitor consistency**: Compare with GR00T training if applicable

For questions or issues, refer to:
- OpenPI docs: `/docs/gr00t_dataset_adapter.md`
- GR00T training docs: `/docs/TRAINING_PIPELINE_GUIDE.md`
- Dataset adapter code: `/src/openpi/training/gr00t_lerobot_dataset.py`
