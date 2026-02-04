#!/usr/bin/env python3
"""
Memory-optimized training script for OpenPI with GR00T dataset.

This script provides several memory optimization strategies to reduce GPU VRAM usage
when training on large datasets with multiple video views.

Usage:
    # Ultra low memory mode (single view, batch size 1)
    python train_memory_optimized.py --mode ultra --dataset-path /path/to/dataset --modality-config-path /path/to/config.py
    
    # Balanced mode (all views with compression, batch size 4)
    python train_memory_optimized.py --mode balanced --dataset-path /path/to/dataset --modality-config-path /path/to/config.py
    
    # Custom configuration
    python train_memory_optimized.py --mode custom --dataset-path /path/to/dataset --modality-config-path /path/to/config.py --batch-size 2 --single-view
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from memory_optimized_training_config import (
    create_memory_optimized_config,
    get_ultra_low_memory_config,
    get_balanced_memory_config,
)


def setup_environment_for_memory_optimization():
    """Set up environment variables for memory optimization."""
    
    # JAX memory settings
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    
    # Reduce JAX compilation cache to save memory
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
    
    # PyTorch settings for data loading
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Reduce OpenMP threads to save memory
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    
    logging.info("Environment configured for memory optimization")


def get_memory_optimized_command_args(config, additional_args=None):
    """Generate command line arguments for memory-optimized training."""
    
    args = [
        "--enable_memory_optimization",
        "--enable_mixed_precision", 
        "--optimization_level", "conservative",
        "--target_memory_utilization", str(config.target_memory_utilization),
        "--jit_warmup_iterations", str(config.jit_warmup_iterations),
        "--batch_size", str(config.batch_size),
        "--num_workers", str(config.num_workers),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--save_interval", str(config.save_interval),
        "--log_interval", str(config.log_interval),
        "--exp_name", config.exp_name,
    ]
    
    if config.overwrite:
        args.append("--overwrite")
    
    # Add data-specific arguments
    if hasattr(config.data, 'dataset_path'):
        args.extend(["--data.dataset_path", config.data.dataset_path])
    if hasattr(config.data, 'embodiment_tag'):
        args.extend(["--data.embodiment_tag", config.data.embodiment_tag])
    if hasattr(config.data, 'modality_config_path'):
        args.extend(["--data.modality_config_path", config.data.modality_config_path])
    if hasattr(config.data, 'video_views') and config.data.video_views:
        for view in config.data.video_views:
            args.extend(["--data.video_views", view])
    
    # Add memory optimization flags
    if hasattr(config.data, 'gr00t_enable_memory_optimization'):
        if config.data.gr00t_enable_memory_optimization:
            args.append("--data.gr00t_enable_memory_optimization")
    
    if additional_args:
        args.extend(additional_args)
    
    return args


def run_training_with_config(config, additional_args=None, dry_run=False):
    """Run training with the given configuration."""
    
    # Set up environment
    setup_environment_for_memory_optimization()
    
    # Generate command arguments
    cmd_args = get_memory_optimized_command_args(config, additional_args)
    
    # Import and run training
    if dry_run:
        print("Would run training with arguments:")
        print(" ".join(cmd_args))
        return
    
    try:
        # Import training modules
        from openpi.training import config as train_config
        from scripts.train import main
        
        # Create a temporary config and add it to the registry
        temp_config = config
        train_config.CONFIGS.append(temp_config)
        
        # Run training
        logging.info(f"Starting memory-optimized training: {config.name}")
        logging.info(f"Batch size: {config.batch_size}, Workers: {config.num_workers}")
        logging.info(f"Video views: {config.data.video_views}")
        logging.info(f"Memory optimization enabled: {getattr(config.data, 'gr00t_enable_memory_optimization', False)}")
        
        main(config)
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized training for OpenPI with GR00T")
    
    # Required arguments
    parser.add_argument("--dataset-path", required=True, help="Path to GR00T dataset")
    parser.add_argument("--modality-config-path", required=True, help="Path to modality config file")
    
    # Mode selection
    parser.add_argument("--mode", choices=["ultra", "balanced", "custom"], default="balanced",
                       help="Memory optimization mode")
    
    # Custom mode arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (custom mode)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers (custom mode)")
    parser.add_argument("--enable-compression", action="store_true", default=True, 
                       help="Enable video frame compression (custom mode)")
    parser.add_argument("--compression-quality", type=int, default=80, 
                       help="JPEG compression quality 0-100 (custom mode)")
    parser.add_argument("--preserve-model-config", action="store_true", default=True,
                       help="Preserve original model configuration (action horizon, views)")
    
    # General arguments
    parser.add_argument("--embodiment-tag", default="dvrk", help="Embodiment tag")
    parser.add_argument("--exp-name", help="Experiment name")
    parser.add_argument("--num-train-steps", type=int, default=30000, help="Number of training steps")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without running")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Additional training arguments
    parser.add_argument("--additional-args", nargs="*", help="Additional arguments to pass to training")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Generate experiment name if not provided
    if not args.exp_name:
        args.exp_name = f"{args.mode}_memory_opt_{args.embodiment_tag}"
    
    # Create configuration based on mode
    if args.mode == "ultra":
        config = get_ultra_low_memory_config(
            args.dataset_path,
            args.modality_config_path,
            args.embodiment_tag,
            args.exp_name,
        )
        logging.info("Using ultra low memory configuration")
        
    elif args.mode == "balanced":
        config = get_balanced_memory_config(
            args.dataset_path,
            args.modality_config_path,
            args.embodiment_tag,
            args.exp_name,
        )
        logging.info("Using balanced memory configuration")
        
    else:  # custom mode
        video_views = ["endoscope_left", "wrist_left", "wrist_right"]  # Always use all 3 views
        
        config = create_memory_optimized_config(
            dataset_path=args.dataset_path,
            modality_config_path=args.modality_config_path,
            embodiment_tag=args.embodiment_tag,
            video_views=video_views,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            exp_name=args.exp_name,
            num_train_steps=args.num_train_steps,
            preserve_model_config=args.preserve_model_config,
        )
        
        # Apply custom compression settings
        if hasattr(config.data, 'gr00t_video_frame_compression'):
            object.__setattr__(config.data, 'gr00t_video_frame_compression', args.enable_compression)
        if hasattr(config.data, 'gr00t_video_frame_quality'):
            object.__setattr__(config.data, 'gr00t_video_frame_quality', args.compression_quality)
        
        logging.info("Using custom memory configuration")
    
    # Print configuration summary
    print("\n" + "="*60)
    print("MEMORY-OPTIMIZED TRAINING CONFIGURATION")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Embodiment: {args.embodiment_tag}")
    print(f"Experiment: {args.exp_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Workers: {config.num_workers}")
    print(f"Video views: {config.data.video_views}")
    print(f"Action horizon: {config.model.action_horizon}")
    print(f"Target memory utilization: {config.target_memory_utilization}")
    
    if hasattr(config.data, 'gr00t_enable_memory_optimization'):
        print(f"Memory optimization: {config.data.gr00t_enable_memory_optimization}")
        print(f"Video compression: {getattr(config.data, 'gr00t_video_frame_compression', False)}")
        print(f"Video cache size: {getattr(config.data, 'gr00t_max_video_cache_size_mb', 'default')}MB")
        print(f"Episode cache size: {getattr(config.data, 'gr00t_episode_cache_size', 'default')}")
    
    print("="*60)
    
    if args.dry_run:
        print("\nDry run mode - configuration created but training not started")
        return
    
    # Confirm before starting
    try:
        response = input("\nStart training with this configuration? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled")
            return
    except KeyboardInterrupt:
        print("\nTraining cancelled")
        return
    
    # Run training
    try:
        run_training_with_config(config, args.additional_args, dry_run=False)
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()