#!/usr/bin/env python3
"""
Memory-optimized training configuration for OpenPI with GR00T dataset.

This script provides optimized configurations to reduce GPU VRAM usage
when training on large datasets with multiple video views.
"""

import dataclasses
from typing import Optional

from openpi.training import config as _config
from openpi.training.config import (
    AssetsConfig, 
    Gr00tLocalLeRobotDataConfig,
    TrainConfig,
)
from openpi.models import pi0
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class MemoryOptimizedGr00tDataConfig(Gr00tLocalLeRobotDataConfig):
    """Memory-optimized GR00T dataset configuration."""
    
    # Memory optimization settings
    gr00t_enable_memory_optimization: bool = True
    gr00t_max_video_cache_size_mb: int = 128  # Reduced from default 256MB
    gr00t_video_frame_compression: bool = True  # Enable JPEG compression
    gr00t_video_frame_quality: int = 80  # JPEG quality (lower = more compression)
    gr00t_lazy_video_loading: bool = True  # Only load videos when needed
    gr00t_reduce_video_resolution: bool = True  # Downsample videos
    gr00t_target_video_resolution: tuple[int, int] = (224, 224)  # Target resolution
    gr00t_enable_frame_skipping: bool = False  # Skip frames (reduces temporal resolution)
    gr00t_frame_skip_factor: int = 1  # Skip every N frames
    
    # Reduced cache sizes
    gr00t_episode_cache_size: int = 4  # Reduced from default 16
    
    # Optimized video backend settings
    gr00t_video_backend: str = "torchcodec"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Override video backend kwargs for memory efficiency
        if self.gr00t_video_backend == "torchcodec":
            video_kwargs = {
                'num_threads': 2,  # Reduced thread count
                'thread_type': 'FRAME',
                'pixel_format': 'rgb24',
                'hardware_acceleration': False,  # Disable to save memory
            }
            object.__setattr__(self, 'gr00t_video_backend_kwargs', video_kwargs)


def create_memory_optimized_config(
    dataset_path: str,
    modality_config_path: str,
    embodiment_tag: str = "dvrk",
    video_views: Optional[list[str]] = None,
    batch_size: int = 2,  # Very small batch size
    num_workers: int = 2,  # Reduced workers
    exp_name: str = "memory_optimized_training",
    num_train_steps: int = 30000,
    preserve_model_config: bool = True,  # Keep original model architecture
) -> TrainConfig:
    """
    Create a memory-optimized training configuration.
    
    Args:
        dataset_path: Path to the GR00T dataset
        modality_config_path: Path to the modality config file
        embodiment_tag: Embodiment tag (e.g., "dvrk")
        video_views: List of video views to use (None for all 3 views)
        batch_size: Batch size (keep small for memory efficiency)
        num_workers: Number of data loading workers
        exp_name: Experiment name
        num_train_steps: Number of training steps
        preserve_model_config: If True, keep original action horizon and image config
        
    Returns:
        Memory-optimized TrainConfig
    """
    
    # Always use all 3 video views to preserve model architecture
    if video_views is None:
        video_views = [
            "endoscope_left",
            "wrist_left", 
            "wrist_right",
        ]
    
    # Create repack transforms for all 3 views (required by model)
    repack_mapping = {
        "state": "observation.state",
        "actions": "actions", 
        "prompt": "prompt",
        "image": {
            "base_0_rgb": "observation.images.endoscope_left",
            "left_wrist_0_rgb": "observation.images.wrist_left",
            "right_wrist_0_rgb": "observation.images.wrist_right",
        }
    }
    
    # Create memory-optimized data config
    data_config = MemoryOptimizedGr00tDataConfig(
        repo_id="local/gr00t_lerobot_memory_optimized",
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        modality_config_path=modality_config_path,
        language_key=None,
        video_views=video_views,  # Keep all 3 views
        base_config=_config.DataConfig(
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(repack_mapping)
                ]
            ),
        ),
    )
    
    # Use original model config to preserve architecture
    if preserve_model_config:
        model_config = pi0.Pi0Config(
            pi05=True,
            # Keep original action horizon (50)
            # Keep original max token length (200)
        )
    else:
        model_config = pi0.Pi0Config(
            pi05=True,
            action_horizon=32,  # Reduced from default 50
            max_token_len=128,  # Reduced from default 200
        )
    
    return TrainConfig(
        name=f"pi05_gr00t_memory_optimized_{embodiment_tag}",
        model=model_config,
        data=data_config,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        # Memory optimization settings
        enable_memory_optimization=True,
        target_memory_utilization=0.6,  # Conservative memory usage
        enable_gradient_accumulation=True,
        gradient_accumulation_steps=max(8, 16 // batch_size),  # Maintain reasonable effective batch size
        # Disable expensive optimizations
        jit_warmup_iterations=1,  # Minimal JIT warmup
        enable_mixed_precision=True,  # Use mixed precision to save memory
        # Checkpoint settings
        save_interval=2000,
        log_interval=50,
        # Experiment settings
        exp_name=exp_name,
        overwrite=True,
    )


def get_ultra_low_memory_config(
    dataset_path: str,
    modality_config_path: str,
    embodiment_tag: str = "dvrk",
    exp_name: str = "ultra_low_memory_training",
) -> TrainConfig:
    """
    Get an ultra-low memory configuration for training on very limited GPU memory.
    
    This configuration uses:
    - Batch size of 1
    - All 3 camera views (preserved for model compatibility)
    - Heavy video compression
    - Minimal caching
    - Original model parameters (action horizon 50, all views)
    """
    
    return create_memory_optimized_config(
        dataset_path=dataset_path,
        modality_config_path=modality_config_path,
        embodiment_tag=embodiment_tag,
        video_views=["endoscope_left", "wrist_left", "wrist_right"],  # Keep all 3 views
        batch_size=1,  # Minimal batch size
        num_workers=1,  # Single worker
        exp_name=exp_name,
        preserve_model_config=True,  # Keep original action horizon and views
    )


def get_balanced_memory_config(
    dataset_path: str,
    modality_config_path: str,
    embodiment_tag: str = "dvrk",
    exp_name: str = "balanced_memory_training",
) -> TrainConfig:
    """
    Get a balanced memory configuration that trades some performance for memory efficiency.
    
    This configuration uses:
    - Batch size of 4
    - All 3 camera views (preserved for model compatibility)
    - Moderate compression and caching
    - Original model parameters (action horizon 50, all views)
    """
    
    return create_memory_optimized_config(
        dataset_path=dataset_path,
        modality_config_path=modality_config_path,
        embodiment_tag=embodiment_tag,
        video_views=["endoscope_left", "wrist_left", "wrist_right"],  # Keep all 3 views
        batch_size=4,
        num_workers=4,
        exp_name=exp_name,
        preserve_model_config=True,  # Keep original action horizon and views
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate memory-optimized training configurations")
    parser.add_argument("--dataset-path", required=True, help="Path to GR00T dataset")
    parser.add_argument("--modality-config-path", required=True, help="Path to modality config file")
    parser.add_argument("--embodiment-tag", default="dvrk", help="Embodiment tag")
    parser.add_argument("--mode", choices=["ultra", "balanced"], default="balanced", 
                       help="Memory optimization mode")
    parser.add_argument("--exp-name", help="Experiment name")
    
    args = parser.parse_args()
    
    if args.mode == "ultra":
        config = get_ultra_low_memory_config(
            args.dataset_path,
            args.modality_config_path,
            args.embodiment_tag,
            args.exp_name or "ultra_low_memory_training",
        )
    else:
        config = get_balanced_memory_config(
            args.dataset_path,
            args.modality_config_path,
            args.embodiment_tag,
            args.exp_name or "balanced_memory_training",
        )
    
    print("Memory-optimized configuration created:")
    print(f"  Name: {config.name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Num workers: {config.num_workers}")
    print(f"  Action horizon: {config.model.action_horizon}")
    print(f"  Video views: {config.data.video_views}")
    print(f"  Memory optimization: {config.data.gr00t_enable_memory_optimization}")
    print(f"  Video compression: {config.data.gr00t_video_frame_compression}")
    print(f"  Target memory utilization: {config.target_memory_utilization}")