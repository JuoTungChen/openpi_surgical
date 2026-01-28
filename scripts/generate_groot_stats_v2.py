#!/usr/bin/env python
"""
Generate GR00T percentile statistics for a dataset (Version 2).

This version uses GR00T's StateActionProcessor to apply transformations
before computing statistics, ensuring the statistics match the training pipeline.

Usage:
    python scripts/generate_groot_stats_v2.py \\
        --dataset-path /path/to/dataset \\
        --embodiment-tag dvrk \\
        --modality-config-path /path/to/modality_config.py
"""

import json
import logging
from pathlib import Path
import importlib.util

import numpy as np
import tyro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_percentile_stats(
    dataset_path: str,
    embodiment_tag: str,
    modality_config_path: str,
    output_path: str | None = None,
    max_samples: int = 10000,
):
    """Compute percentile statistics for a dataset using GR00T's transformation pipeline.
    
    Args:
        dataset_path: Path to LeRobot-format dataset
        embodiment_tag: Embodiment tag (e.g., 'dvrk')
        modality_config_path: Path to GR00T modality config Python file
        output_path: Output path for stats JSON (default: dataset_path/meta/percentile_stats.json)
        max_samples: Maximum number of samples to use for statistics (default: 10000)
    """
    logger.info(f"Computing percentile statistics for: {dataset_path}")
    logger.info(f"Embodiment: {embodiment_tag}")
    logger.info(f"Max samples: {max_samples}")
    
    dataset_path = Path(dataset_path)
    
    # Load modality config
    config_path = Path(modality_config_path)
    spec = importlib.util.spec_from_file_location("modality_config", config_path)
    if not spec or not spec.loader:
        raise ValueError(f"Could not load modality config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get modality config from module
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    if embodiment_tag not in MODALITY_CONFIGS:
        raise ValueError(
            f"Embodiment '{embodiment_tag}' not found in MODALITY_CONFIGS. "
            f"Available: {list(MODALITY_CONFIGS.keys())}"
        )
    
    modality_config = MODALITY_CONFIGS[embodiment_tag]
    logger.info(f"Loaded modality config for '{embodiment_tag}'")
    
    # Load dataset info
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Load episodes
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    
    logger.info(f"Found {len(episodes)} episodes")
    
    # Get data path pattern
    data_path_pattern = info["data_path"]
    chunk_size = info.get("chunks_size", 1000)
    
    state_keys = list(modality_config["state"].modality_keys)
    action_keys = list(modality_config["action"].modality_keys)
    action_configs = modality_config["action"].action_configs
    
    logger.info(f"State keys: {state_keys}")
    logger.info(f"Action keys: {action_keys}")
    
    # Collect RAW state and action data
    all_raw_states = {key: [] for key in state_keys}
    all_raw_actions = {key: [] for key in action_keys}
    
    import pandas as pd
    
    samples_collected = 0
    target_samples = max_samples
    
    logger.info(f"Collecting {target_samples} samples...")
    
    for ep_idx, episode in enumerate(episodes):
        if samples_collected >= target_samples:
            break
        
        if ep_idx % 5 == 0:
            logger.info(f"  Episode {ep_idx}/{len(episodes)}, collected {samples_collected} samples")
        
        episode_index = episode["episode_index"]
        episode_chunk = episode_index // chunk_size
        
        # Load parquet file
        data_path = dataset_path / data_path_pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        
        if not data_path.exists():
            continue
        
        try:
            df = pd.read_parquet(data_path)
        except Exception as e:
            logger.warning(f"Failed to read {data_path}: {e}")
            continue
        
        # Check format
        use_concatenated = "observation.state" in df.columns and "action" in df.columns
        
        # Sample some rows from this episode
        num_rows = min(len(df), (target_samples - samples_collected) // len(episodes) + 1)
        step = max(1, len(df) // num_rows)
        
        for row_idx in range(0, len(df), step):
            if samples_collected >= target_samples:
                break
            
            if use_concatenated:
                # Split concatenated vectors
                state_vec = np.array(df.iloc[row_idx]["observation.state"])
                action_vec = np.array(df.iloc[row_idx]["action"])
                
                # Split by key dimensions
                offset = 0
                for key in state_keys:
                    dim = 7 if "pose" in key.lower() else 1
                    all_raw_states[key].append(state_vec[offset:offset+dim])
                    offset += dim
                
                offset = 0
                for key in action_keys:
                    dim = 7 if "pose" in key.lower() else 1
                    all_raw_actions[key].append(action_vec[offset:offset+dim])
                    offset += dim
            else:
                # Split format
                for key in state_keys:
                    for col in [f"observation.state.{key}", f"observation.{key}", key]:
                        if col in df.columns:
                            all_raw_states[key].append(np.array(df.iloc[row_idx][col]))
                            break
                
                for key in action_keys:
                    for col in [f"action.{key}", key]:
                        if col in df.columns:
                            all_raw_actions[key].append(np.array(df.iloc[row_idx][col]))
                            break
            
            samples_collected += 1
    
    logger.info(f"Collected {samples_collected} samples")
    
    logger.info(f"Collected {samples_collected} samples")
    
    # Apply transformations to actions
    logger.info("Applying action transformations...")
    from gr00t.data.state_action.pose import convert_to_hybrid_relative
    from gr00t.data.types import ActionRepresentation, ActionType, ActionFormat
    
    all_transformed_actions = {}
    
    for idx, key in enumerate(action_keys):
        if action_configs and idx < len(action_configs):
            config = action_configs[idx]
            
            if config.rep == ActionRepresentation.HYBRID_RELATIVE:
                logger.info(f"  {key}: Applying hybrid-relative transformation")
                
                # Get reference states
                state_key = config.state_key if config.state_key else key
                if state_key not in all_raw_states:
                    logger.warning(f"State key '{state_key}' not found, skipping transformation for '{key}'")
                    all_transformed_actions[key] = all_raw_actions[key]
                    continue
                
                states = np.array(all_raw_states[state_key])  # (N, 7)
                actions = np.array(all_raw_actions[key])      # (N, 7)
                
                transformed = []
                for i in range(len(actions)):
                    # Single action with reference state
                    action_step = actions[i:i+1]  # (1, 7)
                    ref_state = states[i]          # (7,)
                    
                    # Apply transformation
                    action_transformed = convert_to_hybrid_relative(
                        action_data=action_step,
                        eef_pose=ref_state,
                        input_rotation_format=config.input_rotation_format or "quat",
                        reference_rotation_format=config.reference_rotation_format or "quat",
                    )
                    
                    # For EEF XYZ_ROT6D, only keep xyz for statistics
                    if config.type == ActionType.EEF and config.format == ActionFormat.XYZ_ROT6D:
                        # action_transformed is (1, 9): xyz_rel (3) + rot6d_rel (6)
                        # Only keep xyz_rel (first 3 dims)
                        transformed.append(action_transformed[0, :3])
                    else:
                        transformed.append(action_transformed[0])
                
                all_transformed_actions[key] = transformed
                logger.info(f"    Transformed {len(transformed)} samples to shape {transformed[0].shape}")
            else:
                # No transformation
                logger.info(f"  {key}: No transformation (keeping raw format)")
                all_transformed_actions[key] = all_raw_actions[key]
        else:
            # No config
            logger.info(f"  {key}: No config (keeping raw format)")
            all_transformed_actions[key] = all_raw_actions[key]
    
    # Compute statistics
    logger.info("Computing percentile statistics...")
    stats = {}
    percentiles = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
    
    # State statistics
    stats["state"] = {}
    for key in state_keys:
        if len(all_raw_states[key]) == 0:
            logger.warning(f"No state data collected for key '{key}'")
            continue
        
        data = np.array(all_raw_states[key])  # (N, dim)
        logger.info(f"  State '{key}': {data.shape}")
        
        stats["state"][key] = {
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
        }
        for p in percentiles:
            stats["state"][key][f"q{p:02d}"] = np.percentile(data, p, axis=0).tolist()
    
    # Action statistics (on TRANSFORMED data)
    stats["action"] = {}
    for key in action_keys:
        if len(all_transformed_actions[key]) == 0:
            logger.warning(f"No action data collected for key '{key}'")
            continue
        
        data = np.array(all_transformed_actions[key])  # (N, dim)
        logger.info(f"  Action '{key}': {data.shape}")
        
        stats["action"][key] = {
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
        }
        for p in percentiles:
            stats["action"][key][f"q{p:02d}"] = np.percentile(data, p, axis=0).tolist()
    
    # Save statistics
    if output_path is None:
        output_path = dataset_path / "meta" / "percentile_stats.json"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Wrap in embodiment_tag key
    output_stats = {embodiment_tag: stats}
    
    with open(output_path, 'w') as f:
        json.dump(output_stats, f, indent=2)
    
    logger.info(f"✓ Saved statistics to: {output_path}")
    logger.info(f"✓ Statistics computed for {len(stats['state'])} state keys and {len(stats['action'])} action keys")
    
    return True


if __name__ == "__main__":
    success = tyro.cli(compute_percentile_stats)
    exit(0 if success else 1)
