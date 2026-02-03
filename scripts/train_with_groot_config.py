#!/usr/bin/env python
"""
Train OpenPI model using GR00T configuration.

This script demonstrates how to train an OpenPI model (π0, π0.5, or π0-FAST)
using GR00T's data loading infrastructure and configurations. This ensures
identical training conditions between GR00T and OpenPI for fair comparison.

Usage:
    # Basic usage with defaults
    python train_with_groot_config.py \\
        --groot-yaml /path/to/gr00t/examples/dVRK/dVRK_multi_config.yaml \\
        --groot-modality /path/to/gr00t/examples/dVRK/dVRK_config.py \\
        --embodiment dvrk

    # Customize model and training
    python train_with_groot_config.py \\
        --groot-yaml /path/to/config.yaml \\
        --groot-modality /path/to/modality_config.py \\
        --embodiment cmr_versius \\
        --model-type pi0_5 \\
        --exp-name my_openpi_experiment \\
        --max-token-len 768

    # Train on a different dataset from the GR00T config
    python train_with_groot_config.py \\
        --groot-yaml /path/to/config.yaml \\
        --groot-modality /path/to/modality_config.py \\
        --embodiment dvrk \\
        --override-dataset-path /path/to/my/dataset \\
        --exp-name custom_dataset_experiment

Features:
    - Automatically loads action_horizon, action_dim from GR00T modality config
    - Uses GR00T's action representations (RELATIVE/ABSOLUTE/HYBRID_RELATIVE)
    - Applies GR00T's normalization statistics
    - Reuses GR00T's training hyperparameters (LR, warmup, batch size)
    - Supports all OpenPI model types (π0, π0.5, π0-FAST)
"""

import logging
from pathlib import Path
from typing import Optional

import tyro

from openpi.models import pi0, pi0_fast
from openpi.models.model import ModelType
from openpi.training.gr00t_config_loader import load_gr00t_config


def main(
    groot_yaml: str,
    groot_modality: str,
    embodiment: str,
    exp_name: str = "openpi_from_groot",
    model_type: str = "pi0_5",
    max_token_len: int = 512,
    override_dataset_path: Optional[str] = None,
    project_name: str = "openpi",
    checkpoint_base_dir: str = "./checkpoints",
    assets_base_dir: str = "./assets",
    log_level: str = "INFO",
):
    """Train OpenPI model using GR00T configuration.

    Args:
        groot_yaml: Path to GR00T YAML config file
            (e.g., "examples/dVRK/dVRK_multi_config.yaml")
        groot_modality: Path to GR00T modality config Python file
            (e.g., "examples/dVRK/dVRK_config.py")
        embodiment: Embodiment tag to train on (must be in modality config)
            (e.g., "dvrk", "cmr_versius", "unitree_g1")
        exp_name: Experiment name for checkpoints and W&B logging
        model_type: OpenPI model type ("pi0", "pi0_5", or "pi0_fast")
        max_token_len: Maximum token length for text encoder
        override_dataset_path: If provided, overrides the dataset path from
            GR00T config (useful for testing on different datasets)
        project_name: W&B project name
        checkpoint_base_dir: Base directory for saving checkpoints
        assets_base_dir: Base directory for assets (norm stats, etc.)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate paths
    groot_yaml_path = Path(groot_yaml)
    if not groot_yaml_path.exists():
        raise FileNotFoundError(f"GR00T YAML config not found: {groot_yaml_path}")

    groot_modality_path = Path(groot_modality)
    if not groot_modality_path.exists():
        raise FileNotFoundError(f"GR00T modality config not found: {groot_modality_path}")

    # Load GR00T configuration
    logger.info("=" * 60)
    logger.info("Loading GR00T Configuration")
    logger.info("=" * 60)
    logger.info(f"YAML config: {groot_yaml_path}")
    logger.info(f"Modality config: {groot_modality_path}")
    logger.info(f"Target embodiment: {embodiment}")

    bundle = load_gr00t_config(
        yaml_path=groot_yaml_path,
        modality_config_path=groot_modality_path,
        load_statistics=True,
    )

    # Validate embodiment exists
    if embodiment not in bundle.modality_configs:
        available = list(bundle.modality_configs.keys())
        raise ValueError(
            f"Embodiment '{embodiment}' not found in modality configs.\n"
            f"Available embodiments: {available}"
        )

    # Extract embodiment settings
    action_horizon = bundle.get_action_horizon(embodiment)
    action_dim = bundle.get_action_dim(embodiment)
    video_views = bundle.get_video_views(embodiment)

    logger.info(f"\nEmbodiment Settings:")
    logger.info(f"  Action horizon: {action_horizon}")
    logger.info(f"  Action dimension: {action_dim}")
    logger.info(f"  Video views: {video_views}")

    # Override dataset path if requested
    if override_dataset_path is not None:
        logger.info(f"\nOverriding dataset path to: {override_dataset_path}")
        # Modify the first dataset config
        if len(bundle.dataset_configs) > 0:
            bundle.dataset_configs[0].dataset_paths[0] = override_dataset_path
        else:
            raise ValueError("No datasets in GR00T config to override")

    # Create model configuration
    logger.info("=" * 60)
    logger.info("Creating Model Configuration")
    logger.info("=" * 60)
    logger.info(f"Model type: {model_type}")

    if model_type.lower() in ["pi0", "pi0_5"]:
        model_config = pi0.Pi0Config(
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_token_len=max_token_len,
            model_type=ModelType.PI05 if model_type.lower() == "pi0_5" else ModelType.PI0,
        )
    elif model_type.lower() == "pi0_fast":
        model_config = pi0_fast.Pi0FastConfig(
            action_horizon=action_horizon,
            action_dim=action_dim,
            max_token_len=max_token_len,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Expected one of: pi0, pi0_5, pi0_fast"
        )

    logger.info(f"Model config:")
    logger.info(f"  Action horizon: {model_config.action_horizon}")
    logger.info(f"  Action dim: {model_config.action_dim}")
    logger.info(f"  Max token length: {model_config.max_token_len}")

    # Convert to OpenPI training config
    logger.info("=" * 60)
    logger.info("Converting to OpenPI Training Config")
    logger.info("=" * 60)

    openpi_config = bundle.to_openpi_config(
        model_config=model_config,
        exp_name=exp_name,
        project_name=project_name,
        checkpoint_base_dir=checkpoint_base_dir,
        assets_base_dir=assets_base_dir,
    )

    # Log dataset information (data is a DataConfigFactory, access factory properties)
    logger.info(f"\nDataset Configuration:")
    if hasattr(openpi_config.data, 'dataset_path'):
        logger.info(f"  Dataset path: {openpi_config.data.dataset_path}")
    if hasattr(openpi_config.data, 'embodiment_tag'):
        logger.info(f"  Embodiment tag: {openpi_config.data.embodiment_tag}")
    if hasattr(openpi_config.data, 'video_backend'):
        logger.info(f"  Video backend: {openpi_config.data.video_backend}")

    # Log training parameters
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Max steps: {openpi_config.num_train_steps}")
    logger.info(f"  Batch size: {openpi_config.batch_size}")
    logger.info(f"  Learning rate: {openpi_config.lr_schedule.peak_lr}")
    logger.info(f"  Warmup steps: {openpi_config.lr_schedule.warmup_steps}")
    logger.info(f"  Weight decay: {openpi_config.optimizer.weight_decay}")

    # Log checkpoint configuration
    logger.info(f"\nCheckpoint Configuration:")
    logger.info(f"  Experiment name: {exp_name}")
    logger.info(f"  Checkpoint dir: {openpi_config.checkpoint_dir}")
    logger.info(f"  Log every: {openpi_config.log_interval} steps")
    logger.info(f"  Checkpoint every: {openpi_config.save_interval} steps")

    # Save GR00T statistics to assets if available
    if bundle.statistics is not None:
        stats_path = Path(assets_base_dir) / embodiment / "groot_statistics.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        bundle.save_statistics(stats_path)
        logger.info(f"\nSaved GR00T statistics to: {stats_path}")

    # Start training
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    # Import here to avoid circular imports
    from openpi.scripts.train import run_training

    try:
        run_training(openpi_config)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        raise

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
