# GR00T Action Representation Transforms

## Overview

The GR00T-OpenPI integration supports two modes for action representation:

1. **Raw Actions** (default fallback if no statistics): Actions remain in their original format from the dataset
2. **Transformed Actions** (with statistics): Actions are transformed using GR00T's action representation (hybrid-relative, rot6d, etc.)

## Action Formats

### Raw Format (16D for dVRK)

Without GR00T statistics, actions are returned in their original dataset format:

```
PSM1_pose:    xyz (3D) + quaternion_xyzw (4D) = 7D
PSM1_gripper: jaw_angle (1D)                  = 1D
PSM2_pose:    xyz (3D) + quaternion_xyzw (4D) = 7D  
PSM2_gripper: jaw_angle (1D)                  = 1D
                                        Total = 16D
```

These are **absolute** positions in world frame.

### Transformed Format (20D for dVRK with Hybrid-Relative)

With GR00T statistics and action representation config, actions are transformed:

```
PSM1_pose_rel:    xyz_rel (3D) + rot6d_rel (6D) = 9D
PSM1_gripper:     jaw_angle (1D)                = 1D  
PSM2_pose_rel:    xyz_rel (3D) + rot6d_rel (6D) = 9D
PSM2_gripper:     jaw_angle (1D)                = 1D
                                          Total = 20D
```

Where:
- `xyz_rel` = translation relative to current EEF position
- `rot6d_rel` = rotation relative to current EEF orientation (6D continuous representation)
- Gripper actions remain absolute

## Generating Statistics

To enable action representation transforms, you need to generate percentile statistics:

```bash
cd /home/iulian/chole_ws/src/openpi

PYTHONPATH=/home/iulian/chole_ws/src/openpi/src:/home/iulian/chole_ws/src/gr00t_n1.6 \
uv run python scripts/generate_groot_stats.py \
  --dataset-path /home/iulian/chole_ws/data/open_h_suturing \
  --embodiment-tag dvrk \
  --modality-config /home/iulian/chole_ws/src/gr00t_n1.6/examples/dVRK/dVRK_config.py
```

This will create `/home/iulian/chole_ws/data/open_h_suturing/meta/percentile_stats.json`.

## Configuration

### Enabling/Disabling Transforms

In `Gr00tLocalLeRobotDataConfig`:

```python
data_config = Gr00tLocalLeRobotDataConfig(
    dataset_path="/path/to/dataset",
    embodiment_tag="dvrk",
    apply_action_transforms=True,  # Enable transforms (default)
    stats_key="dvrk",  # Optional: override stats key
)
```

Set `apply_action_transforms=False` to use raw actions even if statistics are available.

## Model Configuration

**Important**: Your model's `action_dim` must match the format you're using:

### For Raw Actions:
```python
model_config = Pi0Config(
    action_horizon=50,
    action_dim=16,  # Raw format
    max_token_len=512,
)
```

### For Transformed Actions:
```python
model_config = Pi0Config(
    action_horizon=50,
    action_dim=20,  # Transformed format
    max_token_len=512,
)
```

## Automatic Dimension Selection

The `gr00t_config_loader` automatically selects the correct action dimension:

- If `apply_action_transforms=True` and statistics exist → uses `get_processed_action_dim()` (20D)
- Otherwise → uses `get_action_dim()` (16D raw)

## Benefits of Action Representation Transforms

1. **Relative Actions**: Easier for the model to learn (relative to current state)
2. **Rot6D Representation**: Continuous rotation representation (better than quaternions for learning)
3. **Normalization**: Actions are normalized using percentile statistics
4. **Consistency with GR00T**: Matches exactly how GR00T trains its models

## Validation

Check which format is being used:

```bash
PYTHONPATH=/home/iulian/chole_ws/src/openpi/src:/home/iulian/chole_ws/src/gr00t_n1.6 \
uv run scripts/validate_groot_integration.py \
  --groot-yaml ../gr00t_n1.6/examples/dVRK/dVRK_multi_config_test.yaml \
  --groot-modality ../gr00t_n1.6/examples/dVRK/dVRK_config.py \
  --embodiment dvrk
```

Look for:
- Warning about missing statistics → Raw format (16D)
- No warning → Transformed format (20D)
- "Action shape" in Step 5 shows actual dimension

## Troubleshooting

### Actions are 16D but expected 20D

**Cause**: Statistics file not found  
**Solution**: Run `generate_groot_stats.py` script

### Actions are 20D but expected 16D

**Cause**: Statistics exist but you want raw actions  
**Solution**: Set `apply_action_transforms=False` in config

### Wrong action dimension for custom embodiment

**Cause**: `get_action_dim()` or `get_processed_action_dim()` inference failed  
**Solution**: Manually set `action_dim` in model config

## Example: Complete Workflow

```bash
# 1. Generate statistics
PYTHONPATH=$PWD/src:$GROOT_PATH \
uv run python scripts/generate_groot_stats.py \
  --dataset-path /path/to/dataset \
  --embodiment-tag dvrk \
  --modality-config $GROOT_PATH/examples/dVRK/dVRK_config.py

# 2. Validate (should now use 20D actions)
PYTHONPATH=$PWD/src:$GROOT_PATH \
uv run scripts/validate_groot_integration.py \
  --groot-yaml $GROOT_PATH/examples/dVRK/dVRK_multi_config_test.yaml \
  --groot-modality $GROOT_PATH/examples/dVRK/dVRK_config.py \
  --embodiment dvrk

# 3. Train with transformed actions
PYTHONPATH=$PWD/src:$GROOT_PATH \
uv run scripts/train_with_groot_config.py \
  --groot-yaml $GROOT_PATH/examples/dVRK/dVRK_multi_config_test.yaml \
  --groot-modality $GROOT_PATH/examples/dVRK/dVRK_config.py \
  --embodiment dvrk \
  --exp-name openpi_dvrk_hybrid_relative
```

Now your OpenPI model will train with the same hybrid-relative action representation as GR00T! 🎉
