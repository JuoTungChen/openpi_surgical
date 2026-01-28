"""
GR00T Configuration Loader for OpenPI Training

This module enables OpenPI to import and use GR00T's training configurations,
including:
- Modality configs (action representations, delta_indices, action configs)
- Dataset configs (dataset paths, mix ratios, embodiment tags)
- Normalization statistics (per-dataset or shared)
- Action processing settings (RELATIVE/ABSOLUTE/HYBRID_RELATIVE representations)

By reusing GR00T's configs, you ensure that OpenPI training uses identical:
- Action horizons and representations
- State/action key mappings
- Normalization parameters
- Video views and data sampling strategies

This is critical for fair comparison and model compatibility between the two frameworks.

Usage:
    # 1. Load GR00T config from YAML
    from openpi.training.gr00t_config_loader import load_gr00t_config
    
    groot_config = load_gr00t_config(
        yaml_path="examples/dVRK/dVRK_multi_config.yaml",
        modality_config_path="examples/dVRK/dVRK_config.py"
    )
    
    # 2. Convert to OpenPI config
    openpi_config = groot_config.to_openpi_config(
        model_config=pi0.Pi0Config(...),
        exp_name="openpi_dvr_experiment"
    )
    
    # 3. Use in training
    train_state = init_train_state(openpi_config, ...)
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from gr00t.configs.base_config import Config as Gr00tConfig
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.types import ActionConfig, ActionRepresentation, ModalityConfig
except ImportError as e:
    raise ImportError(
        "Failed to import gr00t modules. Please install gr00t:\n"
        "  pip install -e /path/to/gr00t_n1.6\n"
        "or add gr00t to PYTHONPATH."
    ) from e

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms


logger = logging.getLogger(__name__)


@dataclass
class Gr00tConfigBundle:
    """Bundle of GR00T configuration objects for OpenPI integration.
    
    This class holds all the configuration data loaded from GR00T's YAML config
    and modality config files, providing methods to convert them to OpenPI-compatible
    formats.
    
    Attributes:
        gr00t_config: Full GR00T configuration object
        modality_configs: Per-embodiment modality configurations
        dataset_configs: List of dataset configurations with paths and mix ratios
        statistics: Optional normalization statistics
        config_dir: Directory containing the YAML config (for resolving relative paths)
        modality_config_path: Optional path to modality config Python file
    """
    
    gr00t_config: Gr00tConfig
    modality_configs: dict[str, dict[str, ModalityConfig]]
    dataset_configs: list[Any]  # List[SingleDatasetConfig]
    statistics: dict[str, Any] | None
    config_dir: Path
    modality_config_path: Path | None = None
    
    def get_action_horizon(self, embodiment_tag: str) -> int:
        """Get action horizon for an embodiment from its modality config.
        
        Args:
            embodiment_tag: Embodiment identifier (e.g., "dvrk", "cmr_versius")
            
        Returns:
            Action horizon (length of delta_indices in action modality config)
            
        Raises:
            KeyError: If embodiment not found in modality configs
            ValueError: If action modality not configured
        """
        if embodiment_tag not in self.modality_configs:
            available = list(self.modality_configs.keys())
            raise KeyError(
                f"Embodiment '{embodiment_tag}' not found in modality configs. "
                f"Available: {available}"
            )
        
        modality_config = self.modality_configs[embodiment_tag]
        if "action" not in modality_config:
            raise ValueError(
                f"No 'action' modality config for embodiment '{embodiment_tag}'"
            )
        
        action_config = modality_config["action"]
        return len(action_config.delta_indices)
    
    def get_action_dim(self, embodiment_tag: str) -> int:
        """Compute total action dimension from action configs.
        
        IMPORTANT: Returns the RAW action dimension from the dataset, NOT the processed
        dimension after GR00T's action representation transformations.
        
        For example, dVRK has:
        - Raw dataset: 16D (psm1: xyz+quat+jaw=8, psm2: xyz+quat+jaw=8)
        - After HYBRID_RELATIVE + XYZ_ROT6D transform: 20D (psm1: xyz+rot6d+jaw=10, psm2: xyz+rot6d+jaw=10)
        
        OpenPI models should use the RAW dimension (16D) since the dataset adapter
        returns raw actions, and OpenPI applies its own normalization/transforms.
        
        This method attempts to infer the raw dimension from modality_keys first,
        falling back to parsing ActionConfig if needed.
        
        Args:
            embodiment_tag: Embodiment identifier
            
        Returns:
            Raw action dimension from dataset (before GR00T processing)
        """
        modality_config = self.modality_configs[embodiment_tag]
        action_config = modality_config["action"]
        
        # Strategy 1: Try to infer from state modality keys (for robots where action = state format)
        if "state" in modality_config:
            state_config = modality_config["state"]
            state_keys = list(state_config.modality_keys)
            action_keys = list(action_config.modality_keys)
            
            # If action and state have the same keys, they likely have the same dimension
            if set(state_keys) == set(action_keys):
                logger.info(
                    f"Action keys match state keys for {embodiment_tag}, "
                    f"inferring action_dim from known robot configurations"
                )
                # For dVRK: psm1_pose(7) + psm1_gripper(1) + psm2_pose(7) + psm2_gripper(1) = 16
                # Count pose keys (7D each with quat) and gripper keys (1D each)
                pose_count = sum(1 for k in action_keys if 'pose' in k.lower() and 'gripper' not in k.lower())
                gripper_count = sum(1 for k in action_keys if 'gripper' in k.lower() or 'jaw' in k.lower())
                
                # Assume pose keys are xyz+quat=7D, gripper keys are 1D
                inferred_dim = pose_count * 7 + gripper_count * 1
                logger.info(f"Inferred: {pose_count} poses (7D each) + {gripper_count} grippers (1D each) = {inferred_dim}D")
                return inferred_dim
        
        # Strategy 2: Parse ActionConfig input_rotation_format
        logger.warning(
            f"Could not infer action_dim from state keys for {embodiment_tag}, "
            f"parsing ActionConfig (may be inaccurate)"
        )
        
        total_dim = 0
        for i, action_cfg in enumerate(action_config.action_configs):
            # Check if this is an EEF action with rotation
            has_input_rot_attr = hasattr(action_cfg, 'input_rotation_format')
            input_rot_value = getattr(action_cfg, 'input_rotation_format', None) if has_input_rot_attr else None
            
            # Only treat as EEF with rotation if input_rotation_format is explicitly set to a non-empty value
            is_eef_with_rotation = (
                input_rot_value is not None and 
                input_rot_value != '' and
                str(input_rot_value).lower() not in ['none', 'null']
            )
            
            if is_eef_with_rotation:
                # EEF action with rotation - use input format (before processing)
                input_rot = str(input_rot_value).lower()
                if 'quat' in input_rot:
                    total_dim += 7  # xyz (3) + quaternion (4)
                elif 'rot6d' in input_rot:
                    total_dim += 9  # xyz (3) + rot6d (6)
                elif 'euler' in input_rot:
                    total_dim += 6  # xyz (3) + euler (3)
                else:
                    total_dim += 3  # xyz only
            else:
                # Non-EEF action (gripper, joints, etc.) - typically 1D
                total_dim += 1
        
        if total_dim == 0:
            logger.warning(
                f"Could not infer action_dim for {embodiment_tag}. "
                f"You may need to set it manually in model config."
            )
            total_dim = 1
        
        return total_dim
    
    def get_processed_action_dim(self, embodiment_tag: str) -> int:
        """Compute PROCESSED action dimension after GR00T's action representation transforms.
        
        This is the dimension AFTER applying hybrid-relative, rot6d, and other transformations
        specified in the ActionConfig. Use this when apply_action_transforms=True.
        
        For example, dVRK:
        - Raw: 16D (psm1: xyz+quat+jaw=8, psm2: xyz+quat+jaw=8)
        - Processed: 20D (psm1: xyz_rel+rot6d_rel+jaw=10, psm2: xyz_rel+rot6d_rel+jaw=10)
        
        Args:
            embodiment_tag: Embodiment identifier
            
        Returns:
            Processed action dimension (after GR00T transforms)
        """
        modality_config = self.modality_configs[embodiment_tag]
        action_config = modality_config["action"]
        
        total_dim = 0
        for action_cfg in action_config.action_configs:
            # Get the output format after transformation
            if hasattr(action_cfg, 'format'):
                fmt_str = str(action_cfg.format).lower()
                if 'xyz_rot6d' in fmt_str:
                    # xyz (3) + rot6d (6) = 9
                    total_dim += 9
                elif 'xyz' in fmt_str and 'rot' not in fmt_str:
                    # xyz only
                    total_dim += 3
                elif 'rot6d' in fmt_str:
                    # rot6d only
                    total_dim += 6
                elif 'quat' in fmt_str:
                    # quaternion
                    total_dim += 4
                else:
                    # Default (gripper, joint, etc.)
                    total_dim += 1
            else:
                # Non-EEF action (gripper, joints, etc.) - typically 1D
                total_dim += 1
        
        if total_dim == 0:
            logger.warning(
                f"Could not infer processed action_dim for {embodiment_tag}. "
                f"Falling back to raw action_dim."
            )
            return self.get_action_dim(embodiment_tag)
        
        return total_dim
    
    def get_video_views(self, embodiment_tag: str) -> list[str]:
        """Extract video view names from modality config.
        
        Args:
            embodiment_tag: Embodiment identifier
            
        Returns:
            List of video view names (e.g., ["front", "wrist"])
        """
        modality_config = self.modality_configs[embodiment_tag]
        
        # Try to get video views from observation modality config
        if "observation" in modality_config:
            obs_config = modality_config["observation"]
            if hasattr(obs_config, 'modality_keys'):
                # Filter out non-video keys (typically video keys don't contain "state" or "qpos")
                video_views = [
                    key for key in obs_config.modality_keys
                    if 'state' not in key.lower() and 'qpos' not in key.lower()
                ]
                if len(video_views) > 0:
                    return video_views
        
        # Fallback: try to get from video modality config
        if "video" in modality_config:
            video_config = modality_config["video"]
            if hasattr(video_config, 'modality_keys'):
                return list(video_config.modality_keys)
        
        # If still empty, return None instead of empty list to signal that
        # the caller should use defaults or ask the user
        logger.warning(
            f"Could not find video views for {embodiment_tag}. "
            f"Available modalities: {list(modality_config.keys())}"
        )
        return []
    
    def to_openpi_config(
        self,
        model_config: _model.BaseModelConfig,
        exp_name: str,
        project_name: str = "openpi",
        checkpoint_base_dir: str = "./checkpoints",
        assets_base_dir: str = "./assets",
    ) -> _config.TrainConfig:
        """Convert GR00T config to OpenPI TrainConfig.
        
        This creates a complete OpenPI training configuration that uses GR00T's
        data loading infrastructure while training an OpenPI model.
        
        Args:
            model_config: OpenPI model configuration
            exp_name: Experiment name for checkpoints and W&B
            project_name: W&B project name
            checkpoint_base_dir: Base directory for checkpoints
            assets_base_dir: Base directory for assets (norm stats, etc.)
            
        Returns:
            Complete OpenPI TrainConfig ready for training
        """
        # For multi-dataset training, we'll need to create multiple DataConfig objects
        # For now, create one config per dataset and let OpenPI handle mixing
        # (Note: OpenPI doesn't have built-in dataset mixing like GR00T, so this is simplified)
        
        if len(self.dataset_configs) == 0:
            raise ValueError("No datasets configured in GR00T config")
        
        # Use the first dataset as the primary config
        # TODO: Support multi-dataset mixing in OpenPI
        primary_dataset = self.dataset_configs[0]
        embodiment_tag = primary_dataset.embodiment_tag
        dataset_path = primary_dataset.dataset_paths[0]
        
        logger.info(
            f"Creating OpenPI config for embodiment '{embodiment_tag}' "
            f"from dataset: {dataset_path}"
        )
        
        if len(self.dataset_configs) > 1:
            logger.warning(
                f"GR00T config has {len(self.dataset_configs)} datasets, but OpenPI "
                f"config will only use the first one ({embodiment_tag}). "
                f"Multi-dataset mixing not yet supported."
            )
        
        # Determine which action dimension to use based on transforms
        # By default, apply GR00T's action representation transforms
        apply_action_transforms = True
        if apply_action_transforms:
            # Will use processed dimension (after hybrid-relative, rot6d, etc.)
            logger.info(
                f"GR00T action representation transforms will be applied. "
                f"Using processed action dimension."
            )
        else:
            # Will use raw dimension (xyz + quat + gripper)
            logger.info(
                f"GR00T action representation transforms will NOT be applied. "
                f"Using raw action dimension."
            )
        
        # Create data config using GR00T dataset path
        data_config = _config.Gr00tLocalLeRobotDataConfig(
            repo_id=embodiment_tag,  # Use embodiment as repo_id
            dataset_path=dataset_path,
            embodiment_tag=embodiment_tag,
            modality_config_path=str(self.modality_config_path) if self.modality_config_path else None,
            language_key=None,  # Will be inferred from modality config
            video_views=self.get_video_views(embodiment_tag),
            episode_cache_size=1,
            video_backend=self.gr00t_config.data.video_backend,
            apply_action_transforms=apply_action_transforms,
            stats_key=None,  # Will be inferred from embodiment_tag
        )
        
        # Create optimizer config from GR00T training config
        optimizer = _config._optimizer.AdamW(
            weight_decay=self.gr00t_config.training.weight_decay,
        )
        
        # Create LR schedule from GR00T config
        lr_schedule = _config._optimizer.CosineDecaySchedule(
            peak_lr=self.gr00t_config.training.learning_rate,
            warmup_steps=self.gr00t_config.training.warmup_steps,
            decay_steps=self.gr00t_config.training.max_steps,
        )
        
        # Create TrainConfig
        train_config = _config.TrainConfig(
            name=exp_name,
            exp_name=exp_name,
            project_name=project_name,
            model=model_config,
            data=data_config,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            checkpoint_base_dir=checkpoint_base_dir,
            assets_base_dir=assets_base_dir,
            # Copy over training parameters
            num_train_steps=self.gr00t_config.training.max_steps,
            batch_size=self.gr00t_config.training.global_batch_size,
            log_interval=self.gr00t_config.training.logging_steps,
            save_interval=self.gr00t_config.training.save_steps,
        )
        
        return train_config
    
    def create_repack_transforms(
        self,
        embodiment_tag: str,
        target_image_keys: list[str] | None = None,
    ) -> _transforms.Group:
        """Create repack transforms to map GR00T dataset format to OpenPI format.
        
        GR00T datasets store images as "observation.images.<view>" while OpenPI
        expects keys like "image_primary", "image_wrist", etc.
        
        Args:
            embodiment_tag: Embodiment identifier
            target_image_keys: List of target image keys in OpenPI format
                (e.g., ["image_primary", "image_wrist"]). If None, uses default
                mapping from video views.
                
        Returns:
            Transform group that repacks the dataset format
        """
        video_views = self.get_video_views(embodiment_tag)
        
        if target_image_keys is None:
            # Default mapping: first view -> primary, rest numbered
            target_image_keys = []
            if len(video_views) > 0:
                target_image_keys.append("image_primary")
            for i in range(1, len(video_views)):
                target_image_keys.append(f"image_{i}")
        
        # Create rename transforms
        transforms = []
        for src_view, dst_key in zip(video_views, target_image_keys):
            src_key = f"observation.images.{src_view}"
            transforms.append(_transforms.RenameKeys({src_key: dst_key}))
        
        return _transforms.Group(inputs=transforms)
    
    def save_statistics(self, output_path: str | Path) -> None:
        """Save GR00T normalization statistics to JSON.
        
        Args:
            output_path: Path to save statistics JSON
        """
        if self.statistics is None:
            logger.warning("No statistics to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors/arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, list)):
                return np.array(obj).tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            else:
                return obj
        
        stats_json = convert_for_json(self.statistics)
        
        with open(output_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        logger.info(f"Saved GR00T statistics to {output_path}")


def load_modality_config(modality_config_path: str | Path) -> None:
    """Load and register modality config from Python file.
    
    This dynamically imports a Python module that registers embodiment configs
    into gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS.
    
    Args:
        modality_config_path: Path to Python file containing modality config
            (e.g., "examples/dVRK/dVRK_config.py")
    """
    path = Path(modality_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Modality config not found: {path}")
    
    if path.suffix != ".py":
        raise ValueError(f"Modality config must be a .py file: {path}")
    
    # Add parent directory to sys.path temporarily
    parent_dir = str(path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Import the module (this registers configs as a side effect)
        module_name = path.stem
        importlib.import_module(module_name)
        logger.info(f"Loaded modality config from {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to import modality config from {path}: {e}") from e


def load_gr00t_config(
    yaml_path: str | Path,
    modality_config_path: str | Path | None = None,
    load_statistics: bool = True,
) -> Gr00tConfigBundle:
    """Load GR00T configuration from YAML and optional modality config.
    
    This is the main entry point for loading GR00T configs into OpenPI.
    
    Args:
        yaml_path: Path to GR00T YAML config file
            (e.g., "examples/dVRK/dVRK_multi_config.yaml")
        modality_config_path: Optional path to Python modality config file
            (e.g., "examples/dVRK/dVRK_config.py"). If provided, will be loaded
            and registered before loading the YAML config.
        load_statistics: If True, attempts to load normalization statistics
            from the path specified in the config.
            
    Returns:
        Gr00tConfigBundle containing all loaded configuration data
        
    Example:
        >>> bundle = load_gr00t_config(
        ...     yaml_path="examples/dVRK/dVRK_multi_config.yaml",
        ...     modality_config_path="examples/dVRK/dVRK_config.py"
        ... )
        >>> openpi_config = bundle.to_openpi_config(
        ...     model_config=pi0.Pi0Config(
        ...         action_horizon=bundle.get_action_horizon("dvrk"),
        ...         action_dim=bundle.get_action_dim("dvrk"),
        ...     ),
        ...     exp_name="openpi_from_groot"
        ... )
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"GR00T config YAML not found: {yaml_path}")
    
    # Load modality config if provided
    if modality_config_path is not None:
        load_modality_config(modality_config_path)
    
    # Load GR00T config from YAML
    logger.info(f"Loading GR00T config from {yaml_path}")
    config = get_default_config()
    config = config.load(yaml_path)
    
    # Get modality configs (after loading, so custom configs are registered)
    modality_configs = config.data.modality_configs
    
    # Get dataset configs
    dataset_configs = config.data.datasets
    if len(dataset_configs) == 0:
        raise ValueError(f"No datasets configured in {yaml_path}")
    
    # Load statistics if requested
    statistics = None
    if load_statistics and config.data.percentile_stats_path is not None:
        stats_path = Path(config.data.percentile_stats_path)
        if stats_path.exists():
            logger.info(f"Loading statistics from {stats_path}")
            with open(stats_path, 'r') as f:
                statistics = json.load(f)
        else:
            logger.warning(f"Statistics path not found: {stats_path}")
    
    return Gr00tConfigBundle(
        gr00t_config=config,
        modality_configs=modality_configs,
        dataset_configs=dataset_configs,
        statistics=statistics,
        config_dir=yaml_path.parent,
        modality_config_path=Path(modality_config_path) if modality_config_path else None,
    )


def create_openpi_config_from_gr00t(
    yaml_path: str | Path,
    modality_config_path: str | Path | None,
    model_config: _model.BaseModelConfig,
    exp_name: str,
    project_name: str = "openpi",
    checkpoint_base_dir: str = "./checkpoints",
    assets_base_dir: str = "./assets",
) -> _config.TrainConfig:
    """Convenience function to load GR00T config and convert to OpenPI config.
    
    This is a one-liner equivalent to:
        bundle = load_gr00t_config(...)
        config = bundle.to_openpi_config(...)
    
    Args:
        yaml_path: Path to GR00T YAML config
        modality_config_path: Path to modality config Python file
        model_config: OpenPI model configuration
        exp_name: Experiment name
        project_name: W&B project name
        checkpoint_base_dir: Checkpoint directory
        assets_base_dir: Assets directory
        
    Returns:
        OpenPI TrainConfig ready for training
    """
    bundle = load_gr00t_config(yaml_path, modality_config_path)
    return bundle.to_openpi_config(
        model_config=model_config,
        exp_name=exp_name,
        project_name=project_name,
        checkpoint_base_dir=checkpoint_base_dir,
        assets_base_dir=assets_base_dir,
    )
