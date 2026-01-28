# OpenPI Documentation

## Training Guides

- **[GR00T Training Guide](groot_training_guide.md)**: Complete guide for training OpenPI models using GR00T data loaders and action representation transforms

## Quick Start with GR00T

Train a Pi0.5 model with GR00T action representations:

```bash
# 1. Generate statistics on transformed actions
PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
python scripts/generate_groot_stats_v2.py \
  --dataset-path /path/to/dataset/train \
  --modality-config /path/to/modality_config.py \
  --embodiment your_robot \
  --output /path/to/dataset/meta/percentile_stats.json

# 2. Train
PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
uv run scripts/train.py pi05_gr00t_local \
  --exp-name my_experiment \
  --data.dataset-path /path/to/dataset \
  --data.embodiment-tag your_robot \
  --data.modality-config-path /path/to/modality_config.py \
  --num-train-steps 50000
```

See the [full guide](groot_training_guide.md) for detailed instructions.
