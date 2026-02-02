# OpenPI Parameter Names Quick Reference

This document lists the correct parameter names for OpenPI configuration objects. Use this when creating OpenPI configs from GR00T or other sources.

## CosineDecaySchedule (LR Schedule)

**Location**: `openpi.training.optimizer.CosineDecaySchedule`

### Correct Parameters

```python
from openpi.training.optimizer import CosineDecaySchedule

lr_schedule = CosineDecaySchedule(
    warmup_steps=1000,      # ✅ Correct
    peak_lr=2.5e-5,         # ✅ Correct (NOT init_lr)
    decay_steps=30000,      # ✅ Correct
    decay_lr=2.5e-6,        # ✅ Correct (end value)
)
```

### Common Mistakes

```python
# ❌ WRONG - will cause error
lr_schedule = CosineDecaySchedule(
    init_lr=2.5e-5,  # ❌ No such parameter
)

# ✅ CORRECT
lr_schedule = CosineDecaySchedule(
    peak_lr=2.5e-5,  # ✅ Use peak_lr instead
)
```

---

## TrainConfig

**Location**: `openpi.training.config.TrainConfig`

### Correct Parameters

```python
from openpi.training.config import TrainConfig

config = TrainConfig(
    name="my_config",
    exp_name="my_experiment",
    project_name="openpi",
    
    # Model and data
    model=model_config,
    data=data_config,
    
    # Optimization
    optimizer=optimizer_config,
    lr_schedule=lr_schedule_config,
    
    # Training parameters
    num_train_steps=30000,    # ✅ Correct (total steps)
    batch_size=32,            # ✅ Correct (global batch size)
    
    # Logging and checkpointing
    log_interval=100,         # ✅ Correct (NOT log_every_steps)
    save_interval=1000,       # ✅ Correct (NOT checkpoint_every_steps)
    keep_period=5000,         # ✅ Correct (optional)
    
    # Directories
    checkpoint_base_dir="./checkpoints",  # ✅ Correct
    assets_base_dir="./assets",           # ✅ Correct
    
    # Other
    seed=42,                  # ✅ Correct
    num_workers=2,            # ✅ Correct
    wandb_enabled=True,       # ✅ Correct
    resume=False,             # ✅ Correct
    overwrite=False,          # ✅ Correct
)
```

### Common Mistakes

```python
# ❌ WRONG - will cause errors
config = TrainConfig(
    log_every_steps=100,         # ❌ No such parameter
    checkpoint_every_steps=1000, # ❌ No such parameter
)

# ✅ CORRECT
config = TrainConfig(
    log_interval=100,    # ✅ Use log_interval instead
    save_interval=1000,  # ✅ Use save_interval instead
)
```

---

## AdamW Optimizer

**Location**: `openpi.training.optimizer.AdamW`

### Correct Parameters

```python
from openpi.training.optimizer import AdamW

optimizer = AdamW(
    b1=0.9,                    # ✅ Correct (beta1)
    b2=0.95,                   # ✅ Correct (beta2)
    eps=1e-8,                  # ✅ Correct (epsilon)
    weight_decay=1e-10,        # ✅ Correct
    clip_gradient_norm=1.0,    # ✅ Correct
)
```

---

## RsqrtDecaySchedule (Alternative LR Schedule)

**Location**: `openpi.training.optimizer.RsqrtDecaySchedule`

### Correct Parameters

```python
from openpi.training.optimizer import RsqrtDecaySchedule

lr_schedule = RsqrtDecaySchedule(
    warmup_steps=1000,    # ✅ Correct
    peak_lr=5e-5,         # ✅ Correct (NOT init_lr)
    timescale=10000,      # ✅ Correct
)
```

---

## DataConfig

**Location**: `openpi.training.config.DataConfig`

### For GR00T Local Datasets

```python
from openpi.training.config import Gr00tLocalLeRobotDataConfig

data_config = Gr00tLocalLeRobotDataConfig(
    repo_id="embodiment_name",           # ✅ Correct (used for asset_id)
    dataset_path="/path/to/dataset",     # ✅ Correct
    embodiment_tag="dvrk",               # ✅ Correct
    language_key=None,                   # ✅ Correct (optional)
    video_views=["camera_0", "camera_1"], # ✅ Correct (optional)
    episode_cache_size=1,                # ✅ Correct
    video_backend="torchcodec",          # ✅ Correct
)
```

---

## Complete Example: GR00T → OpenPI Conversion

```python
from openpi.training.gr00t_config_loader import load_gr00t_config
from openpi.models.pi0 import Pi0Config

# Load GR00T config
bundle = load_gr00t_config(
    yaml_path="config.yaml",
    modality_config_path="modality.py"
)

# Create model config
model_config = Pi0Config(
    action_horizon=bundle.get_action_horizon("dvrk"),
    action_dim=bundle.get_action_dim("dvrk"),
)

# Convert to OpenPI config (uses correct parameter names internally)
openpi_config = bundle.to_openpi_config(
    model_config=model_config,
    exp_name="my_experiment",
)

# The resulting config will have:
# - lr_schedule with peak_lr (not init_lr)
# - log_interval (not log_every_steps)
# - save_interval (not checkpoint_every_steps)
```

---

## Parameter Name Mapping Table

| Concept | GR00T Name | OpenPI Name | Notes |
|---------|-----------|-------------|-------|
| Learning rate | `learning_rate` | `peak_lr` | Peak value during training |
| Log frequency | `logging_steps` | `log_interval` | Steps between logs |
| Save frequency | `save_steps` | `save_interval` | Steps between checkpoints |
| Max steps | `max_steps` | `num_train_steps` | Total training steps |
| Batch size | `global_batch_size` | `batch_size` | Global batch size |
| Weight decay | `weight_decay` | `weight_decay` | Same ✅ |
| Warmup steps | `warmup_steps` | `warmup_steps` | Same ✅ |

---

## Verification Checklist

When creating OpenPI configs, verify you're using:

- [ ] `peak_lr` (not `init_lr`)
- [ ] `log_interval` (not `log_every_steps` or `log_steps`)
- [ ] `save_interval` (not `checkpoint_every_steps` or `save_steps`)
- [ ] `num_train_steps` (not `max_steps`)
- [ ] `batch_size` (not `global_batch_size` or `per_device_batch_size`)

---

## Common Error Messages

### "got an unexpected keyword argument 'init_lr'"

**Problem**: Using `init_lr` instead of `peak_lr`

**Solution**:
```python
# ❌ Wrong
CosineDecaySchedule(init_lr=1e-4)

# ✅ Correct
CosineDecaySchedule(peak_lr=1e-4)
```

### "got an unexpected keyword argument 'log_every_steps'"

**Problem**: Using `log_every_steps` instead of `log_interval`

**Solution**:
```python
# ❌ Wrong
TrainConfig(log_every_steps=100)

# ✅ Correct
TrainConfig(log_interval=100)
```

### "got an unexpected keyword argument 'checkpoint_every_steps'"

**Problem**: Using `checkpoint_every_steps` instead of `save_interval`

**Solution**:
```python
# ❌ Wrong
TrainConfig(checkpoint_every_steps=1000)

# ✅ Correct
TrainConfig(save_interval=1000)
```

---

## Where to Find Parameter Definitions

1. **CosineDecaySchedule**: `src/openpi/training/optimizer.py` (line ~15)
2. **TrainConfig**: `src/openpi/training/config.py` (line ~525)
3. **AdamW**: `src/openpi/training/optimizer.py` (line ~60)

Or use Python help:
```python
from openpi.training.config import TrainConfig
help(TrainConfig)
```

---

**Last Updated**: January 28, 2026  
**Applies To**: OpenPI v1.x
