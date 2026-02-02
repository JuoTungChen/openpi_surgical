# GR00T + OpenPI Integration

This directory contains tools for training OpenPI models using GR00T's data loading infrastructure and configurations.

## Overview

**Goal**: Enable OpenPI to reuse GR00T's exact data loading setup, ensuring identical training conditions for fair model comparison.

**What This Provides**:
- ✅ Load GR00T YAML configs and modality configs
- ✅ Use GR00T's action representations (RELATIVE/ABSOLUTE/HYBRID_RELATIVE)
- ✅ Apply GR00T's normalization statistics
- ✅ Train OpenPI models (π0, π0.5, π0-FAST) on GR00T-prepared data
- ✅ Maintain consistency between GR00T and OpenPI training

## Quick Start

### 1. Install Dependencies

```bash
# Install both GR00T and OpenPI
pip install -e /path/to/gr00t_n1.6
pip install -e /path/to/openpi
```

### 2. Run Training

```bash
python scripts/train_with_groot_config.py \
    --groot-yaml /path/to/gr00t/examples/dVRK/dVRK_multi_config.yaml \
    --groot-modality /path/to/gr00t/examples/dVRK/dVRK_config.py \
    --embodiment dvrk \
    --exp-name my_openpi_experiment
```

### 3. Monitor Training

```bash
# Checkpoints saved to: ./checkpoints/my_openpi_experiment/
# W&B logging: https://wandb.ai/<user>/openpi/runs/<run-id>
```

## File Structure

```
openpi/
├── src/openpi/training/
│   ├── gr00t_config_loader.py     # GR00T config import and conversion
│   ├── gr00t_lerobot_dataset.py   # Dataset adapter (torch → openpi format)
│   └── config.py                  # Gr00tLocalLeRobotDataConfig
│
├── scripts/
│   └── train_with_groot_config.py # Training script using GR00T configs
│
└── docs/
    ├── gr00t_integration_guide.md # Complete integration guide
    ├── gr00t_dataset_adapter.md   # Dataset adapter documentation
    └── INTEGRATION_README.md      # This file
```

## Components

### 1. Config Loader (`gr00t_config_loader.py`)

Loads GR00T configurations and converts them to OpenPI format.

```python
from openpi.training.gr00t_config_loader import load_gr00t_config

# Load GR00T config
bundle = load_gr00t_config(
    yaml_path="examples/dVRK/dVRK_multi_config.yaml",
    modality_config_path="examples/dVRK/dVRK_config.py"
)

# Extract settings
action_horizon = bundle.get_action_horizon("dvrk")  # e.g., 16
action_dim = bundle.get_action_dim("dvrk")          # e.g., 14
video_views = bundle.get_video_views("dvrk")        # e.g., ["front", "wrist"]

# Convert to OpenPI config
openpi_config = bundle.to_openpi_config(
    model_config=pi0.Pi0Config(
        action_horizon=action_horizon,
        action_dim=action_dim,
    ),
    exp_name="my_experiment"
)
```

**Key Classes**:
- `Gr00tConfigBundle`: Holds loaded GR00T config, provides conversion methods
- `load_gr00t_config()`: Main entry point for loading configs
- `create_openpi_config_from_gr00t()`: Convenience function for one-liner conversion

### 2. Dataset Adapter (`gr00t_lerobot_dataset.py`)

Adapts GR00T's dataset loader to OpenPI's expected format.

```python
from openpi.training.gr00t_lerobot_dataset import Gr00tLeRobotTorchDataset, Gr00tDatasetSpec

# Create dataset
spec = Gr00tDatasetSpec(
    dataset_path="/path/to/lerobot_dataset",
    embodiment_tag="dvrk",
    action_horizon=16,
)
dataset = Gr00tLeRobotTorchDataset(spec)

# Sample data (returns OpenPI-compatible format)
sample = dataset[0]
# {
#   "observation.images.front": [H,W,3],
#   "observation.images.wrist": [H,W,3],
#   "observation.state": [state_dim],
#   "actions": [horizon, action_dim],
#   "prompt": str
# }
```

**Features**:
- Uses GR00T's modality configs (action representations, delta_indices)
- Applies GR00T's StateActionProcessor (normalization, representation conversion)
- Handles clutch-aware filtering (for surgical robots)
- Supports all video backends (torchcodec, pyav, decord)

### 3. Training Script (`train_with_groot_config.py`)

Complete training script that ties everything together.

```bash
# Basic usage
python scripts/train_with_groot_config.py \
    --groot-yaml /path/to/config.yaml \
    --groot-modality /path/to/modality.py \
    --embodiment dvrk

# Customize model
python scripts/train_with_groot_config.py \
    --groot-yaml /path/to/config.yaml \
    --groot-modality /path/to/modality.py \
    --embodiment dvrk \
    --model-type pi0_fast \
    --max-token-len 768

# Override dataset
python scripts/train_with_groot_config.py \
    --groot-yaml /path/to/config.yaml \
    --groot-modality /path/to/modality.py \
    --embodiment dvrk \
    --override-dataset-path /path/to/my/dataset
```

## Configuration Mapping

### GR00T Config → OpenPI Config

| Source | GR00T Config | OpenPI Equivalent |
|--------|-------------|-------------------|
| **Data** | `data.datasets[0].dataset_paths[0]` | `data.gr00t_dataset_path` |
| | `data.datasets[0].embodiment_tag` | `data.gr00t_embodiment_tag` |
| | `data.video_backend` | `data.gr00t_video_backend` |
| | `data.percentile_stats_path` | Loaded via `load_statistics` |
| **Model** | `modality_configs[tag]["action"].delta_indices` | `model.action_horizon` |
| | Sum of action_configs dimensions | `model.action_dim` |
| **Training** | `training.learning_rate` | `lr_schedule.init_lr` |
| | `training.warmup_steps` | `lr_schedule.warmup_steps` |
| | `training.max_steps` | `num_train_steps` |
| | `training.global_batch_size` | `batch_size` |
| | `training.weight_decay` | `optimizer.weight_decay` |

### Action Representations

GR00T's action representations are preserved:

| GR00T Type | Description | OpenPI Support |
|-----------|-------------|----------------|
| `RELATIVE` | Deltas from current state | ✅ Full support |
| `ABSOLUTE` | Target positions | ✅ Full support |
| `HYBRID_RELATIVE` | Relative translation + absolute rotation | ✅ Full support |
| `DELTA` | Incremental changes | ✅ Full support |

The dataset adapter uses GR00T's `StateActionProcessor` internally, so all representation conversions and normalizations are handled correctly.

## Usage Examples

### Example 1: Train π0 on dVRK Dataset

```python
from openpi.training.gr00t_config_loader import load_gr00t_config
from openpi.models.pi0 import Pi0Config
from openpi.scripts.train import run_training

# Load GR00T config
bundle = load_gr00t_config(
    yaml_path="gr00t/examples/dVRK/dVRK_multi_config.yaml",
    modality_config_path="gr00t/examples/dVRK/dVRK_config.py"
)

# Create OpenPI model config
model_config = Pi0Config(
    action_horizon=bundle.get_action_horizon("dvrk"),
    action_dim=bundle.get_action_dim("dvrk"),
    max_token_len=512,
)

# Convert and train
openpi_config = bundle.to_openpi_config(
    model_config=model_config,
    exp_name="pi0_dvrk"
)
run_training(openpi_config)
```

### Example 2: Train π0-FAST on CMR Versius

```python
from openpi.training.gr00t_config_loader import load_gr00t_config
from openpi.models.pi0_fast import Pi0FastConfig

bundle = load_gr00t_config(
    yaml_path="gr00t/examples/CMR_Versius/CMR_multi_config.yaml",
    modality_config_path="gr00t/examples/CMR_Versius/CMR_config.py"
)

model_config = Pi0FastConfig(
    action_horizon=bundle.get_action_horizon("cmr_versius"),
    action_dim=bundle.get_action_dim("cmr_versius"),
)

openpi_config = bundle.to_openpi_config(
    model_config=model_config,
    exp_name="pi0_fast_cmr"
)
run_training(openpi_config)
```

### Example 3: Multi-Embodiment Training

For multiple embodiments, train sequentially:

```python
bundle = load_gr00t_config(yaml_path="...", modality_config_path="...")

for dataset_config in bundle.dataset_configs:
    embodiment = dataset_config.embodiment_tag
    
    model_config = Pi0Config(
        action_horizon=bundle.get_action_horizon(embodiment),
        action_dim=bundle.get_action_dim(embodiment),
    )
    
    openpi_config = bundle.to_openpi_config(
        model_config=model_config,
        exp_name=f"pi0_{embodiment}"
    )
    
    run_training(openpi_config)
```

## Testing Data Loading

Before full training, verify data loads correctly:

```python
from openpi.training.data_loader import create_torch_dataset

# Create dataset
dataset = create_torch_dataset(
    data_config=openpi_config.data,
    action_horizon=openpi_config.model.action_horizon,
    model_config=openpi_config.model,
)

# Test sampling
sample = dataset[0]
print("Keys:", sample.keys())
print("Action shape:", sample["actions"].shape)
print("Expected:", (openpi_config.model.action_horizon, openpi_config.model.action_dim))

# Verify shapes match
assert sample["actions"].shape[0] == openpi_config.model.action_horizon
assert sample["actions"].shape[1] == openpi_config.model.action_dim
```

## Troubleshooting

### ImportError: Failed to import gr00t

**Solution**: Install GR00T
```bash
pip install -e /path/to/gr00t_n1.6
```

### KeyError: Embodiment not found

**Solution**: Check embodiment tag spelling and ensure modality config is loaded
```python
bundle = load_gr00t_config(yaml_path="...", modality_config_path="...")
print("Available embodiments:", list(bundle.modality_configs.keys()))
```

### Action dimension mismatch

**Solution**: Manually specify action_dim
```python
# Instead of automatic detection
action_dim = bundle.get_action_dim("dvrk")

# Use manual value
action_dim = 14  # Set based on your knowledge of the robot
```

### Multi-dataset mixing not working

**Solution**: OpenPI doesn't have built-in mixing. Train sequentially or implement custom data loader.

## Documentation

- **[Integration Guide](docs/gr00t_integration_guide.md)**: Complete guide with examples
- **[Dataset Adapter](docs/gr00t_dataset_adapter.md)**: Technical details of the dataset adapter
- **[GR00T Training Pipeline](../gr00t_n1.6/docs/TRAINING_PIPELINE_GUIDE.md)**: How GR00T training works
- **[Action Representations](../gr00t_n1.6/docs/ACTION_REPRESENTATION_QUICKREF.md)**: Quick reference

## Contributing

When adding new features:

1. Ensure GR00T config compatibility is maintained
2. Test with multiple embodiments (dVRK, CMR, etc.)
3. Update documentation
4. Add examples to this README

## FAQ

**Q: Do I need GR00T installed to use OpenPI?**  
A: Only if you want to use GR00T configs. Native OpenPI training doesn't require GR00T.

**Q: Can I train on HuggingFace LeRobot datasets?**  
A: Yes, use OpenPI's native `LeRobotDataConfig` instead of `Gr00tLocalLeRobotDataConfig`.

**Q: Will this work with future GR00T updates?**  
A: The integration uses GR00T's public APIs. Major changes to GR00T's config structure may require updates.

**Q: Can I mix GR00T and OpenPI datasets?**  
A: Not directly. You'd need to implement custom dataset mixing logic.

**Q: How do I know if action representations are correct?**  
A: The dataset adapter uses GR00T's `StateActionProcessor` internally, which handles all conversions. If GR00T training works, OpenPI training should use the same representations.

## License

See LICENSE file in the root of the OpenPI repository.

## Contact

For issues or questions:
- OpenPI GitHub Issues: https://github.com/physical-intelligence/openpi/issues
- GR00T GitHub Issues: https://github.com/nvidia/gr00t/issues (if applicable)
