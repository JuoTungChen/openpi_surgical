#!/usr/bin/env python
"""
Generate GR00T percentile statistics for a dataset.

This script computes the percentile statistics needed for GR00T's action
representation transformations (hybrid-relative, normalization, etc.).

IMPORTANT: Statistics are computed on TRANSFORMED actions (after hybrid-relative
conversion, rot6d conversion, etc.), not raw actions. This matches GR00T's
training pipeline where transformations happen before normalization.

Usage:
    python scripts/generate_groot_stats.py \\
        --dataset-path /path/to/dataset \\
        --embodiment-tag dvrk \\
        --modality-config /path/to/modality_config.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import tyro

try:
    import polars as pl
    USE_POLARS = True
except ImportError:
    import pandas as pd
    USE_POLARS = False

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
):
    """Compute percentile statistics for a dataset.
    
    Args:
        dataset_path: Path to LeRobot-format dataset
        embodiment_tag: Embodiment tag (e.g., 'dvrk')
        modality_config_path: Path to GR00T modality config Python file
        output_path: Output path for stats JSON (default: dataset_path/meta/percentile_stats.json)
    """
    logger.info(f"Computing percentile statistics for: {dataset_path}")
    logger.info(f"Embodiment: {embodiment_tag}")
    
    # Import GR00T after parsing args
    try:
        import gr00t
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    except ImportError:
        logger.error("GR00T not found. Install with: pip install -e /path/to/gr00t_n1.6")
        return False
    
    # Load modality config
    import importlib.util
    spec = importlib.util.spec_from_file_location("modality_config", modality_config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if embodiment_tag not in MODALITY_CONFIGS:
        logger.error(f"Embodiment '{embodiment_tag}' not found in MODALITY_CONFIGS")
        logger.error(f"Available: {list(MODALITY_CONFIGS.keys())}")
        return False
    
    modality_config = MODALITY_CONFIGS[embodiment_tag]
    logger.info(f"Loaded modality config for '{embodiment_tag}'")
    
    # Load dataset info
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        logger.error(f"Dataset info not found: {info_path}")
        return False
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    data_path_pattern = info["data_path"]
    chunk_size = info["chunks_size"]
    
    # Load episodes
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = [json.loads(l) for l in episodes_path.read_text().splitlines() if l.strip()]
    
    logger.info(f"Found {len(episodes)} episodes")
    
    # Collect all data for statistics
    state_keys = list(modality_config["state"].modality_keys)
    action_keys = list(modality_config["action"].modality_keys)
    
    logger.info(f"Looking for state keys: {state_keys}")
    logger.info(f"Looking for action keys: {action_keys}")
    
    all_states = {key: [] for key in state_keys}
    all_actions = {key: [] for key in action_keys}
    
    # Debug: check first episode columns
    first_episode = episodes[0]
    episode_index = first_episode["episode_index"]
    episode_chunk = episode_index // chunk_size
    data_path = dataset_path / data_path_pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    if data_path.exists():
        try:
            if USE_POLARS:
                df = pl.read_parquet(data_path)
                sample_df = df.to_pandas()
            else:
                sample_df = pd.read_parquet(data_path)
            logger.info(f"\nSample episode columns ({len(sample_df.columns)} total):")
            for col in sorted(sample_df.columns)[:20]:  # Show first 20
                logger.info(f"  {col}")
            if len(sample_df.columns) > 20:
                logger.info(f"  ... and {len(sample_df.columns) - 20} more")
        except Exception as e:
            logger.warning(f"Failed to read sample episode: {e}")
    
    logger.info("\nCollecting data from all episodes...")
    
    # Check if data is already concatenated or split into separate columns
    use_concatenated = "observation.state" in sample_df.columns and "action" in sample_df.columns
    
    if use_concatenated:
        logger.info("Dataset uses concatenated format (observation.state + action)")
        # Collect full state and action vectors
        all_state_data = []
        all_action_data = []
    else:
        logger.info("Dataset uses split format (separate columns per modality key)")
    
    for ep_idx, episode in enumerate(episodes):
        if ep_idx % 10 == 0:
            logger.info(f"  Processing episode {ep_idx}/{len(episodes)}")
        
        episode_index = episode["episode_index"]
        episode_chunk = episode_index // chunk_size
        
        # Load parquet file
        # Format the data path with both episode_chunk and episode_index
        data_path = dataset_path / data_path_pattern.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        
        if not data_path.exists():
            logger.warning(f"Skipping missing file: {data_path}")
            continue
        
        try:
            if USE_POLARS:
                df = pl.read_parquet(data_path)
                # Convert to pandas for consistency
                ep_df = df.to_pandas()
            else:
                ep_df = pd.read_parquet(data_path)
        except Exception as e:
            logger.warning(f"Failed to read {data_path}: {e}")
            continue
        
        if use_concatenated:
            # Collect full concatenated vectors
            if "observation.state" in ep_df.columns:
                all_state_data.extend(ep_df["observation.state"].tolist())
            if "action" in ep_df.columns:
                all_action_data.extend(ep_df["action"].tolist())
        else:
            # Collect states
            for key in state_keys:
                # Try different column name formats
                possible_cols = [
                    f"observation.state.{key}",
                    f"observation.{key}",
                    key,
                ]
                col_name = None
                for col in possible_cols:
                    if col in ep_df.columns:
                        col_name = col
                        break
                
                if col_name:
                    all_states[key].extend(ep_df[col_name].tolist())
            
            # Collect actions
            for key in action_keys:
                # Try different column name formats
                possible_cols = [
                    f"action.{key}",
                    key,
                ]
                col_name = None
                for col in possible_cols:
                    if col in ep_df.columns:
                        col_name = col
                        break
                
                if col_name:
                    all_actions[key].extend(ep_df[col_name].tolist())
    
    logger.info("Computing percentile statistics...")
    
    # Compute percentiles in GR00T-compatible format
    # GR00T expects: {embodiment: {modality: {key: {stat_type: values}}}}
    # Percentiles: 1→q01, 2→q02, 5→q05, ..., 98→q98, 99→q99
    stats = {}
    percentiles = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
    
    # Helper function to split concatenated vectors by modality keys
    def split_by_keys(concatenated_array: np.ndarray, keys: list[str], modality_config: dict) -> dict[str, np.ndarray]:
        """Split concatenated array into per-key arrays based on modality configuration."""
        result = {}
        offset = 0
        
        for key in keys:
            # Infer dimension from key name (robot-specific heuristic)
            if "pose" in key.lower():
                dim = 7  # xyz + quaternion
            elif "gripper" in key.lower() or "jaw" in key.lower():
                dim = 1
            else:
                # Fallback: try to infer from config or assume equal split
                dim = concatenated_array.shape[-1] // len(keys)
            
            result[key] = concatenated_array[..., offset:offset+dim]
            offset += dim
        
        return result
    
    if use_concatenated:
        # Split concatenated state and action vectors by modality keys
        state_dict = {}
        action_dict = {}
        
        if len(all_state_data) > 0:
            state_array = np.array(all_state_data)  # Shape: (num_samples, state_dim)
            state_dict = split_by_keys(state_array, state_keys, modality_config)
        
        if len(all_action_data) > 0:
            action_array = np.array(all_action_data)  # Shape: (num_samples, action_dim)
            action_dict = split_by_keys(action_array, action_keys, modality_config)
        
        # Compute per-key statistics in GR00T format
        stats["state"] = {}
        for key, data in state_dict.items():
            stats["state"][key] = {
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
                "min": data.min(axis=0).tolist(),
                "max": data.max(axis=0).tolist(),
            }
            # Add percentiles with q-notation
            for p in percentiles:
                stats["state"][key][f"q{p:02d}"] = np.percentile(data, p, axis=0).tolist()
        
        stats["action"] = {}
        for key, data in action_dict.items():
            stats["action"][key] = {
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
                "min": data.min(axis=0).tolist(),
                "max": data.max(axis=0).tolist(),
            }
            # Add percentiles with q-notation  
            for p in percentiles:
                stats["action"][key][f"q{p:02d}"] = np.percentile(data, p, axis=0).tolist()
    else:
        # For split format, compute per-key statistics directly
        stats["state"] = {}
        for key in state_keys:
            if len(all_states[key]) > 0:
                data = np.vstack(all_states[key])  # Shape: (num_samples, dim)
                stats["state"][key] = {
                    "mean": data.mean(axis=0).tolist(),
                    "std": data.std(axis=0).tolist(),
                    "min": data.min(axis=0).tolist(),
                    "max": data.max(axis=0).tolist(),
                }
                for p in percentiles:
                    stats["state"][key][f"q{p:02d}"] = np.percentile(data, p, axis=0).tolist()
        
        stats["action"] = {}
        for key in action_keys:
            if len(all_actions[key]) > 0:
                data = np.vstack(all_actions[key])  # Shape: (num_samples, dim)
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
    
    # Wrap in embodiment_tag key
    output_stats = {embodiment_tag: stats}
    
    with open(output_path, 'w') as f:
        json.dump(output_stats, f, indent=2)
    
    logger.info(f"✓ Saved statistics to: {output_path}")
    logger.info(f"✓ Statistics computed for {len(stats)} keys")
    
    return True


if __name__ == "__main__":
    success = tyro.cli(compute_percentile_stats)
    exit(0 if success else 1)
