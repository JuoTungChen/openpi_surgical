#!/usr/bin/env python
"""
Validation script for GR00T-OpenPI integration.

This script validates that GR00T configurations can be loaded correctly
and converted to OpenPI configs without errors. It checks:

1. GR00T config loading
2. Modality config registration
3. Action horizon/dim extraction
4. OpenPI config conversion
5. Dataset instantiation
6. Data sampling

Usage:
    python scripts/validate_groot_integration.py \\
        --groot-yaml /path/to/config.yaml \\
        --groot-modality /path/to/modality.py \\
        --embodiment dvrk

    # Dry-run mode (skip dataset instantiation)
    python scripts/validate_groot_integration.py \\
        --groot-yaml /path/to/config.yaml \\
        --groot-modality /path/to/modality.py \\
        --embodiment dvrk \\
        --dry-run
"""

import logging
from pathlib import Path
from typing import Optional

import tyro

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_imports():
    """Validate that required packages are installed."""
    logger.info("=" * 60)
    logger.info("Step 1: Validating Imports")
    logger.info("=" * 60)
    
    try:
        import gr00t
        logger.info("✓ GR00T installed")
    except ImportError as e:
        logger.error("✗ GR00T not found")
        logger.error(f"  Error: {e}")
        logger.error("  Install with: pip install -e /path/to/gr00t_n1.6")
        return False
    
    try:
        import openpi
        logger.info("✓ OpenPI installed")
    except ImportError as e:
        logger.error("✗ OpenPI not found")
        logger.error(f"  Error: {e}")
        return False
    
    try:
        from openpi.training.gr00t_config_loader import load_gr00t_config
        logger.info("✓ GR00T config loader available")
    except ImportError as e:
        logger.error("✗ GR00T config loader not found")
        logger.error(f"  Error: {e}")
        return False
    
    try:
        from openpi.training.gr00t_lerobot_dataset import Gr00tLeRobotTorchDataset
        logger.info("✓ GR00T dataset adapter available")
    except ImportError as e:
        logger.error("✗ GR00T dataset adapter not found")
        logger.error(f"  Error: {e}")
        return False
    
    logger.info("\n✅ All imports successful\n")
    return True


def validate_config_loading(
    groot_yaml: Path,
    groot_modality: Path,
    embodiment: str,
):
    """Validate GR00T config loading."""
    logger.info("=" * 60)
    logger.info("Step 2: Loading GR00T Configuration")
    logger.info("=" * 60)
    
    from openpi.training.gr00t_config_loader import load_gr00t_config
    
    try:
        bundle = load_gr00t_config(
            yaml_path=groot_yaml,
            modality_config_path=groot_modality,
            load_statistics=True,
        )
        logger.info("✓ GR00T config loaded successfully")
    except Exception as e:
        logger.error("✗ Failed to load GR00T config")
        logger.error(f"  Error: {e}")
        return None
    
    # Validate embodiment exists
    if embodiment not in bundle.modality_configs:
        available = list(bundle.modality_configs.keys())
        logger.error(f"✗ Embodiment '{embodiment}' not found")
        logger.error(f"  Available: {available}")
        return None
    
    logger.info(f"✓ Embodiment '{embodiment}' found")
    
    # Extract settings
    try:
        action_horizon = bundle.get_action_horizon(embodiment)
        logger.info(f"✓ Action horizon: {action_horizon}")
    except Exception as e:
        logger.error(f"✗ Failed to get action horizon: {e}")
        return None
    
    try:
        action_dim = bundle.get_action_dim(embodiment)
        logger.info(f"✓ Action dim: {action_dim}")
    except Exception as e:
        logger.error(f"✗ Failed to get action dim: {e}")
        return None
    
    try:
        video_views = bundle.get_video_views(embodiment)
        logger.info(f"✓ Video views: {video_views}")
    except Exception as e:
        logger.error(f"✗ Failed to get video views: {e}")
        return None
    
    # Log dataset info
    logger.info(f"\nDataset Configuration:")
    logger.info(f"  Number of datasets: {len(bundle.dataset_configs)}")
    for i, ds in enumerate(bundle.dataset_configs):
        logger.info(f"  Dataset {i}:")
        logger.info(f"    Embodiment: {ds.embodiment_tag}")
        logger.info(f"    Path: {ds.dataset_paths[0]}")
        logger.info(f"    Mix ratio: {ds.mix_ratio}")
    
    # Log statistics
    if bundle.statistics is not None:
        logger.info(f"\n✓ Normalization statistics loaded")
        logger.info(f"  Keys: {list(bundle.statistics.keys())[:5]}...")
    else:
        logger.info(f"\n⚠ No normalization statistics found")
    
    logger.info("\n✅ Config loading successful\n")
    return bundle


def validate_openpi_conversion(bundle, embodiment: str):
    """Validate conversion to OpenPI config."""
    logger.info("=" * 60)
    logger.info("Step 3: Converting to OpenPI Config")
    logger.info("=" * 60)
    
    from openpi.models.pi0 import Pi0Config
    
    try:
        model_config = Pi0Config(
            action_horizon=bundle.get_action_horizon(embodiment),
            action_dim=bundle.get_action_dim(embodiment),
            max_token_len=512,
        )
        logger.info("✓ OpenPI model config created")
    except Exception as e:
        logger.error(f"✗ Failed to create model config: {e}")
        return None
    
    try:
        openpi_config = bundle.to_openpi_config(
            model_config=model_config,
            exp_name="validation_test",
        )
        logger.info("✓ OpenPI training config created")
    except Exception as e:
        logger.error(f"✗ Failed to convert to OpenPI config: {e}")
        return None
    
    # Log converted config
    logger.info(f"\nTraining Config:")
    logger.info(f"  Experiment: {openpi_config.exp_name}")
    
    # Access factory properties (data is a DataConfigFactory, not DataConfig)
    if hasattr(openpi_config.data, 'dataset_path'):
        logger.info(f"  Dataset path: {openpi_config.data.dataset_path}")
    if hasattr(openpi_config.data, 'embodiment_tag'):
        logger.info(f"  Embodiment: {openpi_config.data.embodiment_tag}")
    
    logger.info(f"  Action horizon: {openpi_config.model.action_horizon}")
    logger.info(f"  Action dim: {openpi_config.model.action_dim}")
    logger.info(f"  Batch size: {openpi_config.batch_size}")
    logger.info(f"  Learning rate: {openpi_config.lr_schedule.peak_lr}")
    logger.info(f"  Max steps: {openpi_config.num_train_steps}")
    
    logger.info("\n✅ OpenPI conversion successful\n")
    return openpi_config


def validate_dataset_instantiation(openpi_config):
    """Validate dataset can be instantiated."""
    logger.info("=" * 60)
    logger.info("Step 4: Instantiating Dataset")
    logger.info("=" * 60)
    
    from openpi.training.data_loader import create_torch_dataset
    import pathlib
    
    # Create the actual DataConfig from the factory
    try:
        # For validation, we use a dummy assets path
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
    
    try:
        dataset = create_torch_dataset(
            data_config=data_config,
            action_horizon=openpi_config.model.action_horizon,
            model_config=openpi_config.model,
        )
        logger.info("✓ Dataset created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    try:
        dataset_size = len(dataset)
        logger.info(f"✓ Dataset size: {dataset_size} samples")
    except Exception as e:
        logger.error(f"✗ Failed to get dataset size: {e}")
        return None
    
    logger.info("\n✅ Dataset instantiation successful\n")
    return dataset


def validate_data_sampling(dataset, openpi_config, bundle, embodiment):
    """Validate data can be sampled."""
    logger.info("=" * 60)
    logger.info("Step 5: Sampling Data")
    logger.info("=" * 60)
    
    try:
        sample = dataset[0]
        logger.info("✓ Sample retrieved successfully")
    except Exception as e:
        logger.error(f"✗ Failed to sample data: {e}")
        return False
    
    # Validate sample structure
    logger.info(f"\nSample Structure:")
    logger.info(f"  Keys: {list(sample.keys())}")
    
    # Check action shape
    if "actions" in sample:
        action_shape = sample["actions"].shape
        
        # Determine expected action dimension based on whether transforms are applied
        expected_action_dim = openpi_config.model.action_dim
        
        # Check if action transforms are enabled in the data config
        transforms_enabled = False
        # Try both attribute names (factory has apply_action_transforms, DataConfig has gr00t_apply_action_transforms)
        if hasattr(openpi_config.data, 'gr00t_apply_action_transforms'):
            transforms_enabled = openpi_config.data.gr00t_apply_action_transforms
        elif hasattr(openpi_config.data, 'apply_action_transforms'):
            transforms_enabled = openpi_config.data.apply_action_transforms
        
        if transforms_enabled:
            # Use processed action dimension (with transformations)
            try:
                expected_action_dim = bundle.get_processed_action_dim(embodiment)
                logger.info(f"  Action transforms enabled: using processed dimension")
            except Exception:
                # Fall back to raw dimension if processed dimension not available
                pass
        
        expected_shape = (
            openpi_config.model.action_horizon,
            expected_action_dim,
        )
        
        logger.info(f"  Action shape: {action_shape}")
        logger.info(f"  Expected: {expected_shape}")
        
        if transforms_enabled:
            logger.info(f"  Note: Actions are transformed (hybrid-relative, rot6d, etc.)")
        else:
            logger.info(f"  Note: Actions are in raw format (no transformations)")
        
        if action_shape == expected_shape:
            logger.info("  ✓ Action shape matches")
        else:
            logger.error("  ✗ Action shape mismatch!")
            return False
    else:
        logger.error("  ✗ 'actions' key not found in sample")
        return False
    
    # Check for images
    image_keys = [k for k in sample.keys() if "image" in k.lower() or "observation.images" in k]
    if image_keys:
        logger.info(f"  Image keys: {image_keys}")
        for key in image_keys:
            logger.info(f"    {key}: {sample[key].shape}")
        logger.info("  ✓ Images found")
    else:
        logger.warning("  ⚠ No image keys found")
    
    # Check for prompt
    if "prompt" in sample:
        prompt_val = sample['prompt']
        # Handle both string and numpy scalar
        if hasattr(prompt_val, 'item'):
            prompt_str = str(prompt_val.item())
        else:
            prompt_str = str(prompt_val)
        
        # Truncate long prompts
        display_prompt = prompt_str[:50] + '...' if len(prompt_str) > 50 else prompt_str
        logger.info(f"  Prompt: '{display_prompt}'")
        logger.info("  ✓ Prompt found")
    else:
        logger.warning("  ⚠ No prompt found")
    
    logger.info("\n✅ Data sampling successful\n")
    return True


def main(
    groot_yaml: str,
    groot_modality: str,
    embodiment: str,
    dry_run: bool = False,
    override_dataset_path: Optional[str] = None,
):
    """Validate GR00T-OpenPI integration.
    
    Args:
        groot_yaml: Path to GR00T YAML config
        groot_modality: Path to GR00T modality config Python file
        embodiment: Embodiment tag to validate
        dry_run: If True, skip dataset instantiation and sampling
        override_dataset_path: If provided, override the dataset path from GR00T config
    """
    logger.info("GR00T-OpenPI Integration Validation")
    logger.info("=" * 60)
    
    # Validate paths
    groot_yaml_path = Path(groot_yaml)
    groot_modality_path = Path(groot_modality)
    
    if not groot_yaml_path.exists():
        logger.error(f"✗ GR00T YAML not found: {groot_yaml_path}")
        return False
    
    if not groot_modality_path.exists():
        logger.error(f"✗ GR00T modality config not found: {groot_modality_path}")
        return False
    
    logger.info(f"YAML config: {groot_yaml_path}")
    logger.info(f"Modality config: {groot_modality_path}")
    logger.info(f"Embodiment: {embodiment}")
    logger.info(f"Dry run: {dry_run}\n")
    
    # Step 1: Validate imports
    if not validate_imports():
        logger.error("\n❌ Validation failed at Step 1: Imports\n")
        return False
    
    # Step 2: Load GR00T config
    bundle = validate_config_loading(groot_yaml_path, groot_modality_path, embodiment)
    if bundle is None:
        logger.error("\n❌ Validation failed at Step 2: Config Loading\n")
        return False
    
    # Override dataset path if provided
    if override_dataset_path is not None:
        logger.info(f"\nOverriding dataset path to: {override_dataset_path}")
        if len(bundle.dataset_configs) > 0:
            bundle.dataset_configs[0].dataset_paths[0] = override_dataset_path
        else:
            logger.error("No datasets in GR00T config to override")
            return False
    
    # Step 3: Convert to OpenPI config
    openpi_config = validate_openpi_conversion(bundle, embodiment)
    if openpi_config is None:
        logger.error("\n❌ Validation failed at Step 3: OpenPI Conversion\n")
        return False
    
    if dry_run:
        logger.info("=" * 60)
        logger.info("Dry-run mode: Skipping dataset instantiation")
        logger.info("=" * 60)
        logger.info("\n✅ Validation successful (dry-run)\n")
        return True
    
    # Step 4: Instantiate dataset
    dataset = validate_dataset_instantiation(openpi_config)
    if dataset is None:
        logger.error("\n❌ Validation failed at Step 4: Dataset Instantiation\n")
        return False
    
    # Step 5: Sample data
    if not validate_data_sampling(dataset, openpi_config, bundle, embodiment):
        logger.error("\n❌ Validation failed at Step 5: Data Sampling\n")
        return False
    
    # Success!
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info("✅ Step 1: Imports - PASSED")
    logger.info("✅ Step 2: Config Loading - PASSED")
    logger.info("✅ Step 3: OpenPI Conversion - PASSED")
    logger.info("✅ Step 4: Dataset Instantiation - PASSED")
    logger.info("✅ Step 5: Data Sampling - PASSED")
    logger.info("=" * 60)
    logger.info("\n🎉 All validation checks passed! Ready to train.\n")
    
    return True


if __name__ == "__main__":
    success = tyro.cli(main)
    exit(0 if success else 1)
