# GR00T-OpenPI Integration: Bug Fixes

## Fix #1: Incorrect Learning Rate Parameter Name

**Date**: January 28, 2026  
**Severity**: Critical  
**Status**: ✅ Fixed

### Issue

```python
# Error during validation:
CosineDecaySchedule.__init__() got an unexpected keyword argument 'init_lr'
```

### Root Cause

The config loader was using `init_lr` parameter name when creating `CosineDecaySchedule`, but OpenPI's implementation uses `peak_lr` instead.

### Fix

Changed in `gr00t_config_loader.py`:

```python
# Before (incorrect)
lr_schedule = _config._optimizer.CosineDecaySchedule(
    init_lr=self.gr00t_config.training.learning_rate,  # ❌ Wrong parameter name
    warmup_steps=self.gr00t_config.training.warmup_steps,
    decay_steps=self.gr00t_config.training.max_steps,
)

# After (correct)
lr_schedule = _config._optimizer.CosineDecaySchedule(
    peak_lr=self.gr00t_config.training.learning_rate,  # ✅ Correct parameter name
    warmup_steps=self.gr00t_config.training.warmup_steps,
    decay_steps=self.gr00t_config.training.max_steps,
)
```

### Updated Documentation

- `docs/gr00t_integration_guide.md` - Configuration mapping table
- `docs/GROOT_INTEGRATION_SUMMARY.md` - Configuration mapping table

Changed from:
```
| training.learning_rate | lr_schedule.init_lr | Initial learning rate |
```

To:
```
| training.learning_rate | lr_schedule.peak_lr | Peak learning rate |
```

---

## Fix #2: Incorrect TrainConfig Parameter Names

**Date**: January 28, 2026  
**Severity**: Critical  
**Status**: ✅ Fixed

### Issue

```python
# Error during validation:
TrainConfig.__init__() got an unexpected keyword argument 'log_every_steps'
```

### Root Cause

The config loader was using `log_every_steps` and `checkpoint_every_steps` parameter names when creating `TrainConfig`, but OpenPI uses `log_interval` and `save_interval` instead.

### Fix

Changed in `gr00t_config_loader.py`:

```python
# Before (incorrect)
train_config = _config.TrainConfig(
    # ...
    log_every_steps=self.gr00t_config.training.logging_steps,      # ❌ Wrong
    checkpoint_every_steps=self.gr00t_config.training.save_steps,  # ❌ Wrong
)

# After (correct)
train_config = _config.TrainConfig(
    # ...
    log_interval=self.gr00t_config.training.logging_steps,  # ✅ Correct
    save_interval=self.gr00t_config.training.save_steps,    # ✅ Correct
)
```

---

## Fix #3: Improved Video Views Detection

**Date**: January 28, 2026  
**Severity**: Medium  
**Status**: ✅ Fixed

### Issue

```python
# Warning during validation:
Could not find video views for dvrk, returning empty list
```

### Root Cause

The `get_video_views()` method only checked the `observation` modality and used overly aggressive filtering. Some embodiment configs may organize video views differently or not have an `observation` modality.

### Fix

Enhanced `get_video_views()` in `gr00t_config_loader.py`:

```python
def get_video_views(self, embodiment_tag: str) -> list[str]:
    """Extract video view names from modality config."""
    modality_config = self.modality_configs[embodiment_tag]
    
    # Try observation modality first
    if "observation" in modality_config:
        obs_config = modality_config["observation"]
        if hasattr(obs_config, 'modality_keys'):
            video_views = [
                key for key in obs_config.modality_keys
                if 'state' not in key.lower() and 'qpos' not in key.lower()
            ]
            if len(video_views) > 0:
                return video_views
    
    # Fallback: try video modality
    if "video" in modality_config:
        video_config = modality_config["video"]
        if hasattr(video_config, 'modality_keys'):
            return list(video_config.modality_keys)
    
    # More informative warning
    logger.warning(
        f"Could not find video views for {embodiment_tag}. "
        f"Available modalities: {list(modality_config.keys())}"
    )
    return []
```

### Benefits

1. **Fallback mechanism**: Checks `video` modality if `observation` fails
2. **Better logging**: Shows available modalities to help debug
3. **More robust**: Returns empty list gracefully without breaking

---

## Fix #4: Validation Script Logging Issues

**Date**: January 28, 2026  
**Severity**: Minor  
**Status**: ✅ Fixed

### Issue

```python
# Error in validation script:
AttributeError: 'Gr00tLocalLeRobotDataConfig' object has no attribute 'gr00t_dataset_path'
```

### Root Cause

The validation and training scripts were trying to access `openpi_config.data.gr00t_dataset_path`, but `openpi_config.data` is a `DataConfigFactory` (not a `DataConfig`), so it has `dataset_path` not `gr00t_dataset_path`.

Also, the scripts were using old parameter names (`init_lr`, `log_every_steps`, etc.) in logging.

### Fix

Updated both `validate_groot_integration.py` and `train_with_groot_config.py`:

```python
# Before (incorrect)
logger.info(f"  Dataset path: {openpi_config.data.gr00t_dataset_path}")  # ❌
logger.info(f"  Learning rate: {openpi_config.lr_schedule.init_lr}")     # ❌
logger.info(f"  Log every: {openpi_config.log_every_steps} steps")       # ❌

# After (correct)
if hasattr(openpi_config.data, 'dataset_path'):
    logger.info(f"  Dataset path: {openpi_config.data.dataset_path}")    # ✅
logger.info(f"  Learning rate: {openpi_config.lr_schedule.peak_lr}")     # ✅
logger.info(f"  Log every: {openpi_config.log_interval} steps")          # ✅
```

### Benefits

1. **Correct field access**: Uses factory properties correctly
2. **Safe attribute access**: Uses `hasattr()` to avoid errors
3. **Consistent naming**: Uses correct OpenPI parameter names

---

## Testing

After fixes, validation should pass:

```bash
python scripts/validate_groot_integration.py \
    --groot-yaml /path/to/config.yaml \
    --groot-modality /path/to/modality.py \
    --embodiment dvrk \
    --dry-run
```

Expected output:
```
✅ Step 1: Imports - PASSED
✅ Step 2: Config Loading - PASSED
✅ Step 3: OpenPI Conversion - PASSED  # ✅ Now fixed
...
```

---

## Known Limitations

### Video Views Detection

If video views still cannot be detected automatically, you can specify them manually:

```python
bundle = load_gr00t_config(yaml_path="...", modality_config_path="...")

# Override video views manually
openpi_config = bundle.to_openpi_config(model_config=..., exp_name="...")
openpi_config = dataclasses.replace(
    openpi_config,
    data=dataclasses.replace(
        openpi_config.data,
        gr00t_video_views=["camera_0", "camera_1"],  # Your actual view names
    )
)
```

Or in the training script:

```bash
python scripts/train_with_groot_config.py \
    --groot-yaml ... \
    --groot-modality ... \
    --embodiment dvrk
    # Then manually edit the config in your script
```

### Multi-Dataset Training

Still not supported. OpenPI config will only use the first dataset from GR00T's multi-dataset config. This is a known limitation, not a bug.

---

## Version History

### v1.0.2 (2026-01-28)
- ✅ Fixed TrainConfig parameter names (`log_every_steps` → `log_interval`, `checkpoint_every_steps` → `save_interval`)

### v1.0.1 (2026-01-28)
- ✅ Fixed learning rate parameter name (`init_lr` → `peak_lr`)
- ✅ Improved video views detection with fallback
- ✅ Enhanced error messages

### v1.0.0 (2026-01-28)
- Initial release
- Core integration implementation
- Documentation suite

---

## Verification Checklist

Before using the integration, verify:

- [ ] GR00T and OpenPI are installed
- [ ] Validation script passes (Step 1-3 at minimum)
- [ ] Video views are detected or manually specified
- [ ] Dataset path exists and is readable
- [ ] Embodiment tag matches modality config

---

## Fix #5: DataConfigFactory vs DataConfig Confusion

**Date**: January 28, 2026  
**Severity**: **High**  
**Status**: ✅ Fixed

### Issue

```python
# Error in validation script (Step 4):
AttributeError: 'Gr00tLocalLeRobotDataConfig' object has no attribute 'gr00t_dataset_path'
```

### Root Cause

The validation script was passing `openpi_config.data` (a `Gr00tLocalLeRobotDataConfig` factory) directly to `create_torch_dataset()`, which expects an actual `DataConfig` instance.

**Key distinction:**
- `Gr00tLocalLeRobotDataConfig` is a **factory** (extends `DataConfigFactory`)
- `DataConfig` is the **actual config** with all fields populated
- Must call `.create(assets_dirs, model_config)` on factory to get `DataConfig`

The training pipeline correctly handles this conversion in `create_data_loader()`:
```python
# From data_loader.py line 253
def create_data_loader(config: TrainConfig, ...):
    data_config = config.data.create(config.assets_dirs, config.model)  # ✅ Correct
    ...
```

But the validation script was incorrectly calling:
```python
# WRONG - passing factory instead of config
dataset = create_torch_dataset(
    data_config=openpi_config.data,  # ❌ This is a factory, not DataConfig!
    ...
)
```

### Fix

Updated `validate_groot_integration.py` to create `DataConfig` from factory first:

```python
def validate_dataset_instantiation(openpi_config):
    """Validate dataset can be instantiated."""
    from openpi.training.data_loader import create_torch_dataset
    import pathlib
    
    # ✅ Create the actual DataConfig from the factory
    try:
        assets_path = pathlib.Path("/tmp/openpi_validation_assets")
        assets_path.mkdir(parents=True, exist_ok=True)
        
        data_config = openpi_config.data.create(
            assets_dirs=assets_path,
            model_config=openpi_config.model
        )
        logger.info("✓ DataConfig created from factory")
    except Exception as e:
        logger.error(f"✗ Failed to create DataConfig: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ✅ Now pass the actual DataConfig to create_torch_dataset
    try:
        dataset = create_torch_dataset(
            data_config=data_config,  # ✅ Correct - actual DataConfig instance
            action_horizon=openpi_config.model.action_horizon,
            model_config=openpi_config.model,
        )
        logger.info("✓ Dataset created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    ...
```

### Benefits

1. **Correct API usage**: Respects the factory pattern used in OpenPI
2. **Better error messages**: Added traceback printing for debugging
3. **Matches training code**: Uses same pattern as `create_data_loader()`
4. **Validates full pipeline**: Tests both factory→config and config→dataset steps

### Related Files

- `/home/iulian/chole_ws/src/openpi/src/openpi/training/config.py` (line 471): `Gr00tLocalLeRobotDataConfig` factory
- `/home/iulian/chole_ws/src/openpi/src/openpi/training/data_loader.py` (line 253): Correct usage example

---

## Fix #6: Action Dimension Calculation Error

**Date**: January 28, 2026  
**Severity**: **Critical**  
**Status**: ✅ Fixed

### Issue

```
Action shape mismatch!
  Action shape: (50, 16)
  Expected: (50, 28)
```

### Root Cause

The `get_action_dim()` method was incorrectly calculating the action dimension by trying to parse `ActionConfig.input_rotation_format`, but the logic was flawed. It was returning 28D instead of the correct 16D for dVRK.

The problem: ActionConfig parsing is complex and error-prone because:
1. Gripper configs may or may not have `input_rotation_format` attribute
2. The attribute might be `None`, empty string, or other edge cases
3. We need RAW dimensions (before GR00T processing), not processed dimensions

### Fix

Rewrote `get_action_dim()` with a heuristic-based approach:

**Strategy 1** (primary): Infer from state modality keys
```python
# For robots where action format matches state format (common case)
if set(state_keys) == set(action_keys):
    # Count pose keys (7D with quaternion) and gripper keys (1D)
    pose_count = sum(1 for k in action_keys if 'pose' in k.lower())
    gripper_count = sum(1 for k in action_keys if 'gripper' in k.lower())
    return pose_count * 7 + gripper_count * 1  # 16D for dVRK
```

**Strategy 2** (fallback): Parse ActionConfig with safer checks
```python
# Only treat as EEF with rotation if explicitly set
is_eef_with_rotation = (
    input_rot_value is not None and 
    input_rot_value != '' and
    str(input_rot_value).lower() not in ['none', 'null']
)
```

### Benefits

1. **Accurate for dVRK**: Correctly returns 16D (2 poses × 7D + 2 grippers × 1D)
2. **Robust heuristic**: Uses modality key names instead of fragile attribute parsing
3. **Fallback available**: Still parses ActionConfig if heuristic fails
4. **Better logging**: Explains inference strategy

### Additional Fixes

**Fix #6.1**: Prompt display bug in validation script
- **Issue**: `IndexError` when displaying numpy scalar prompts
- **Fix**: Handle both string and numpy scalar types with `.item()` method

**Fix #6.2**: Missing modality.json in dataset
- **Issue**: `FileNotFoundError: meta/modality.json`  
- **Solution**: Copy from GR00T examples to dataset directory:
  ```bash
  cp /path/to/gr00t/examples/dVRK/modality.json /path/to/dataset/meta/
  ```

---

## Reporting Issues

If you encounter issues:

1. **Run validation script**: `python scripts/validate_groot_integration.py --dry-run ...`
2. **Check which step fails**: The script will show exactly where the problem is
3. **Consult troubleshooting**: See `docs/gr00t_integration_guide.md` - Troubleshooting section
4. **Check this file**: See if your issue is a known limitation

---

## Future Improvements

Potential enhancements based on user feedback:

1. **Auto-detect video views from dataset**: Inspect parquet files directly
2. **Better action_dim inference**: Use GR00T's `get_action_dim()` directly
3. **Validation warnings**: Make non-critical warnings more visible
4. **Multi-dataset support**: Implement OpenPI-side dataset mixing

---

**Last Updated**: January 28, 2026  
**Compatibility**: GR00T v1.x, OpenPI v1.x
