# Training OpenPI Models with GR00T Data Loaders

This guide explains how to train OpenPI models (Pi0/Pi0.5) using GR00T's data loaders and action representation transforms.

## Overview

The GR00T integration enables OpenPI to:
- Load data using GR00T's LeRobot dataset loaders
- Apply GR00T's action representation transforms (hybrid-relative coordinates, rot6d)
- Train with the exact same action space used in GR00T deployment

## Prerequisites

1. **GR00T repository**: Clone and install GR00T in the same environment
   ```bash
   git clone <groot-repo-url> /path/to/gr00t_n1.6
   pip install -e /path/to/gr00t_n1.6
   ```

2. **OpenPI repository**: Already set up with this integration
   ```bash
   cd /path/to/openpi
   pip install -e .
   ```

3. **Dataset**: LeRobot-format dataset with GR00T modality config
   - Must have `meta/info.json`, `meta/episodes.jsonl`, and episode parquet files
   - Must have a corresponding GR00T modality config (`.py` file defining `MODALITY_CONFIGS`)

## Workflow

### Step 1: Prepare Your Dataset

Ensure your dataset follows the LeRobot format:
```
your_dataset/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── (percentile_stats.json - generated in step 2)
├── data/
│   └── chunk-*/
│       └── episode_*.parquet
└── videos/
    └── chunk-*/
        └── episode_*_{view}.mp4
```

### Step 2: Generate Statistics for Action Transforms

**Important**: Generate statistics on **transformed** actions (not raw data):

```bash
cd /path/to/openpi

PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
python scripts/generate_groot_stats_v2.py \
  --dataset-path /path/to/your_dataset/train \
  --modality-config /path/to/gr00t_n1.6/examples/YOUR_ROBOT/modality_config.py \
  --embodiment your_embodiment_name \
  --output /path/to/your_dataset/meta/percentile_stats.json
```

**What this does**:
- Loads raw episodes from your dataset
- Applies GR00T's `convert_to_hybrid_relative()` transformation
- Computes percentile statistics (q01-q99) on transformed data
- Saves statistics for use by `StateActionProcessor`

**Example for dVRK**:
```bash
PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
python scripts/generate_groot_stats_v2.py \
  --dataset-path /data/open_h_suturing/train \
  --modality-config /path/to/gr00t_n1.6/examples/dVRK/dVRK_config.py \
  --embodiment dvrk \
  --output /data/open_h_suturing/meta/percentile_stats.json
```

### Step 3: Validate Integration (Optional but Recommended)

Verify that everything is configured correctly:

```bash
PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
uv run scripts/validate_groot_integration.py \
  --groot-yaml /path/to/gr00t_config.yaml \
  --groot-modality /path/to/modality_config.py \
  --embodiment your_embodiment
```

The validation script checks:
- ✅ All required packages are installed
- ✅ GR00T config loads successfully
- ✅ Config converts to OpenPI format correctly
- ✅ Dataset can be instantiated
- ✅ **Action dimensions are correct** (e.g., 20D transformed vs 16D raw)

### Step 4: Configure Training

**Option A: Use CLI Overrides (Recommended)**

Override the default `pi05_gr00t_local` config:

```bash
cd /path/to/openpi

PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
uv run scripts/train.py pi05_gr00t_local \
  --exp-name my_experiment_name \
  --data.dataset-path /path/to/your_dataset \
  --data.embodiment-tag your_embodiment \
  --data.modality-config-path /path/to/modality_config.py \
  --data.video-views view1 view2 view3 \
  --data.apply-action-transforms \
  --num-train-steps 50000 \
  --batch-size 16 \
  --num-workers 4
```

**Option B: Create a New Config**

Edit `src/openpi/training/config.py` and add:

```python
TrainConfig(
    name="my_robot_config",
    model=pi0.Pi0Config(pi05=True),
    data=Gr00tLocalLeRobotDataConfig(
        repo_id="my_org/my_dataset",
        dataset_path="/path/to/your_dataset",
        embodiment_tag="your_embodiment",
        modality_config_path="/path/to/modality_config.py",
        language_key=None,
        video_views=["view1", "view2", "view3"],
        apply_action_transforms=True,  # Enable GR00T transforms
        stats_key=None,  # Will use embodiment_tag by default
        base_config=DataConfig(
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform({
                        "image": {
                            "base_0_rgb": "observation.images.view1",
                            "left_wrist_0_rgb": "observation.images.view2",
                            "right_wrist_0_rgb": "observation.images.view3",
                        },
                        "state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",
                    })
                ]
            ),
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/params"
    ),
    num_train_steps=50_000,
),
```

Then train:
```bash
PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
uv run scripts/train.py my_robot_config \
  --exp-name experiment_001
```

### Step 5: Start Training

Full training command with recommended settings:

```bash
cd /path/to/openpi

PYTHONPATH=/path/to/openpi/src:/path/to/gr00t_n1.6 \
uv run scripts/train.py pi05_gr00t_local \
  --exp-name dvrk_suturing_v1 \
  --data.dataset-path /data/open_h_suturing \
  --data.embodiment-tag dvrk \
  --data.modality-config-path /path/to/gr00t_n1.6/examples/dVRK/dVRK_config.py \
  --data.video-views endoscope_left wrist_left wrist_right \
  --data.apply-action-transforms \
  --num-train-steps 50000 \
  --batch-size 16 \
  --num-workers 4 \
  --save-interval 1000 \
  --log-interval 100 \
  --wandb-enabled
```

## Understanding Action Transformations

### With Action Transforms Enabled (`apply_action_transforms=True`)

**What happens**:
1. Raw actions loaded from dataset (e.g., 16D: xyz+quat+gripper per arm)
2. `StateActionProcessor` applies transformations:
   - Converts to hybrid-relative coordinates
   - Converts quaternions to rot6d representation
   - Normalizes using percentile statistics
3. Model trains on transformed actions (e.g., 20D: xyz_rel+rot6d+gripper per arm)

**Action dimension example (dual-arm dVRK)**:
- Raw: 16D = `[psm1(xyz(3)+quat(4)+gripper(1)) + psm2(xyz(3)+quat(4)+gripper(1))]`
- Transformed: 20D = `[psm1(xyz_rel(3)+rot6d(6)+gripper(1)) + psm2(xyz_rel(3)+rot6d(6)+gripper(1))]`

**Normalization**: Handled by GR00T's `StateActionProcessor` (OpenPI's normalization is automatically disabled)

### With Action Transforms Disabled (`apply_action_transforms=False`)

- Actions remain in raw format from dataset
- OpenPI's normalization can be used (if norm_stats are provided)
- **Not recommended** if you plan to deploy with GR00T

## Configuration Parameters

### Data Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dataset-path` | Path to LeRobot dataset root | `/data/my_dataset` |
| `embodiment-tag` | Embodiment name (must match `MODALITY_CONFIGS` key) | `dvrk` |
| `modality-config-path` | Path to GR00T modality config file | `/path/to/dVRK_config.py` |
| `video-views` | List of video view names (from modality config) | `endoscope_left wrist_left` |
| `apply-action-transforms` | Enable GR00T action transforms | `--data.apply-action-transforms` (default: True) |
| `stats-key` | Key for statistics in percentile_stats.json | `dvrk` (default: uses embodiment-tag) |

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `exp-name` | Experiment name (required) | - |
| `num-train-steps` | Number of training steps | 30000 |
| `batch-size` | Global batch size | 32 |
| `num-workers` | Number of data loader workers | 2 |
| `save-interval` | Checkpoint save interval | 1000 |
| `log-interval` | Logging interval | 100 |
| `wandb-enabled` | Enable Weights & Biases logging | True |

## Troubleshooting

### Issue: "Modality config file not found"

**Solution**: Ensure the path to your modality config is correct:
```bash
ls -la /path/to/modality_config.py
```

### Issue: "Statistics key not found"

**Solution**: Regenerate statistics using `generate_groot_stats_v2.py`. Make sure:
- The embodiment name matches your modality config
- Statistics are generated on the correct split (train vs val)

### Issue: "Action shape mismatch"

**Cause**: Action dimensions don't match expected values.

**Check**:
1. Is `apply_action_transforms` enabled?
2. Are statistics loaded correctly?
3. Run validation script to see expected vs actual dimensions

### Issue: "ValueError: operands could not be broadcast"

**Cause**: OpenPI's normalization is trying to apply wrong-sized statistics.

**Solution**: This should be automatically handled. If you see this error, verify:
- `apply_action_transforms=True` is set
- Statistics were generated with `generate_groot_stats_v2.py` (not the old script)

### Issue: Video decoding errors

**Solution**: Check video backend settings:
```bash
--data.video-backend torchcodec  # or 'pyav', 'decord'
```

## Example: Complete dVRK Training Workflow

```bash
# Setup paths
export OPENPI_PATH="/path/to/openpi"
export GROOT_PATH="/path/to/gr00t_n1.6"
export DATA_PATH="/data/open_h_suturing"
export PYTHONPATH="$OPENPI_PATH/src:$GROOT_PATH"

cd $OPENPI_PATH

# Step 1: Generate statistics
python scripts/generate_groot_stats_v2.py \
  --dataset-path $DATA_PATH/train \
  --modality-config $GROOT_PATH/examples/dVRK/dVRK_config.py \
  --embodiment dvrk \
  --output $DATA_PATH/meta/percentile_stats.json

# Step 2: Validate (optional)
uv run scripts/validate_groot_integration.py \
  --groot-yaml $GROOT_PATH/examples/dVRK/dVRK_multi_config_test.yaml \
  --groot-modality $GROOT_PATH/examples/dVRK/dVRK_config.py \
  --embodiment dvrk

# Step 3: Train
uv run scripts/train.py pi05_gr00t_local \
  --exp-name dvrk_suturing_exp1 \
  --data.dataset-path $DATA_PATH \
  --data.embodiment-tag dvrk \
  --data.modality-config-path $GROOT_PATH/examples/dVRK/dVRK_config.py \
  --data.video-views endoscope_left wrist_left wrist_right \
  --num-train-steps 50000 \
  --batch-size 16 \
  --num-workers 4 \
  --save-interval 1000 \
  --wandb-enabled
```

## What Happens During Training

1. **Data Loading**:
   - `Gr00tLeRobotTorchDataset` loads episodes using GR00T's `LeRobotEpisodeLoader`
   - Video frames decoded using specified backend (torchcodec/pyav/decord)

2. **Action Processing**:
   - Raw actions: xyz+quaternion+gripper per end-effector
   - `StateActionProcessor` converts to hybrid-relative: xyz_rel+rot6d+gripper
   - Percentile normalization applied (q02/q98 clipping)

3. **Model Training**:
   - Pi0.5 model receives transformed actions
   - Action head predicts in transformed space
   - **Consistency**: Training uses exact same representation as GR00T deployment

## Key Benefits

✅ **Consistency**: Train and deploy with identical action representations
✅ **Modular**: Reuses GR00T configs without code duplication  
✅ **Flexible**: Can disable transforms for comparison experiments
✅ **Validated**: Includes validation script to catch configuration errors

## Advanced Topics

### Using Multiple Datasets

Currently, the integration supports single-dataset training. For multi-dataset training, you would need to extend the config to handle multiple `Gr00tLocalLeRobotDataConfig` instances.

### Custom Action Representations

To use custom action transforms:
1. Define new `ActionConfig` types in GR00T
2. Update `StateActionProcessor` to handle new types
3. Generate statistics with the new transforms
4. Train with `apply_action_transforms=True`

### Fine-tuning from Pre-trained Checkpoints

```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "gs://your-bucket/checkpoints/your_model/params"
)
```

## References

- GR00T Documentation: `/path/to/gr00t_n1.6/README.md`
- OpenPI Documentation: `/path/to/openpi/README.md`
- LeRobot Dataset Format: https://github.com/huggingface/lerobot

## Support

For issues or questions:
1. Check validation script output for configuration errors
2. Review logs in `wandb/` directory
3. Verify statistics were generated correctly
4. Compare with example configs in `src/openpi/training/config.py`
