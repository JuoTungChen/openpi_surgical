#!/usr/bin/env python3
"""
Script to check memory requirements before training.
This helps estimate if the current configuration will fit in GPU memory.
"""

import sys
import os
sys.path.insert(0, '/app/src')

import jax
import jax.numpy as jnp
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def check_gpu_memory():
    """Check available GPU memory."""
    try:
        # Get GPU memory info
        devices = jax.devices('gpu')
        if not devices:
            logging.error("No GPU devices found!")
            return False
        
        logging.info(f"Found {len(devices)} GPU device(s)")
        for i, device in enumerate(devices):
            logging.info(f"GPU {i}: {device}")
        
        # Try to allocate a small tensor to check memory
        test_tensor = jnp.ones((1000, 1000), dtype=jnp.float32)
        logging.info(f"Successfully allocated test tensor: {test_tensor.shape}")
        
        return True
    except Exception as e:
        logging.error(f"GPU memory check failed: {e}")
        return False

def estimate_model_memory(batch_size=1):
    """Estimate memory requirements for the model."""
    try:
        # Import the model config
        from openpi.training.config import CONFIGS
        
        # Get the pi05_gr00t_local config
        config = None
        for cfg in CONFIGS:
            if cfg.name == "pi05_gr00t_local":
                config = cfg
                break
        
        if config is None:
            logging.error("Could not find pi05_gr00t_local config")
            return False
        
        logging.info(f"Found config: {config.name}")
        logging.info(f"Model config: {config.model}")
        logging.info(f"Default batch size: {getattr(config, 'batch_size', 'Not set')}")
        
        # Estimate memory based on model parameters
        # Pi0.5 typically has around 3B parameters
        # Each parameter in FP32 = 4 bytes, in FP16 = 2 bytes
        estimated_params = 3_000_000_000  # 3B parameters
        
        # Memory for model weights (FP32)
        model_memory_fp32 = estimated_params * 4 / (1024**3)  # GB
        # Memory for model weights (FP16)
        model_memory_fp16 = estimated_params * 2 / (1024**3)  # GB
        
        # Memory for gradients (same as model weights)
        gradient_memory_fp32 = model_memory_fp32
        gradient_memory_fp16 = model_memory_fp16
        
        # Memory for optimizer states (Adam typically 2x model size)
        optimizer_memory_fp32 = model_memory_fp32 * 2
        optimizer_memory_fp16 = model_memory_fp16 * 2
        
        # Memory for activations (depends on batch size and sequence length)
        # Rough estimate: batch_size * sequence_length * hidden_size * layers * 4 bytes
        sequence_length = 256  # Typical for robotics
        hidden_size = 2048  # Typical for Pi0.5
        num_layers = 24  # Typical for Pi0.5
        
        activation_memory = batch_size * sequence_length * hidden_size * num_layers * 4 / (1024**3)  # GB
        
        total_memory_fp32 = model_memory_fp32 + gradient_memory_fp32 + optimizer_memory_fp32 + activation_memory
        total_memory_fp16 = model_memory_fp16 + gradient_memory_fp16 + optimizer_memory_fp16 + activation_memory
        
        logging.info("=== Memory Estimation ===")
        logging.info(f"Estimated parameters: {estimated_params:,}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Sequence length: {sequence_length}")
        logging.info("")
        logging.info("FP32 Memory Requirements:")
        logging.info(f"  Model weights: {model_memory_fp32:.2f} GB")
        logging.info(f"  Gradients: {gradient_memory_fp32:.2f} GB")
        logging.info(f"  Optimizer states: {optimizer_memory_fp32:.2f} GB")
        logging.info(f"  Activations: {activation_memory:.2f} GB")
        logging.info(f"  Total: {total_memory_fp32:.2f} GB")
        logging.info("")
        logging.info("FP16 Memory Requirements:")
        logging.info(f"  Model weights: {model_memory_fp16:.2f} GB")
        logging.info(f"  Gradients: {gradient_memory_fp16:.2f} GB")
        logging.info(f"  Optimizer states: {optimizer_memory_fp16:.2f} GB")
        logging.info(f"  Activations: {activation_memory:.2f} GB")
        logging.info(f"  Total: {total_memory_fp16:.2f} GB")
        logging.info("")
        
        # Check if it fits in typical A100 memory (40GB or 80GB)
        a100_40gb = 40
        a100_80gb = 80
        
        logging.info("=== Memory Fit Analysis ===")
        if total_memory_fp32 <= a100_40gb:
            logging.info(f"✓ FP32 training should fit in A100-40GB ({total_memory_fp32:.2f} GB <= {a100_40gb} GB)")
        else:
            logging.warning(f"✗ FP32 training may not fit in A100-40GB ({total_memory_fp32:.2f} GB > {a100_40gb} GB)")
        
        if total_memory_fp16 <= a100_40gb:
            logging.info(f"✓ FP16 training should fit in A100-40GB ({total_memory_fp16:.2f} GB <= {a100_40gb} GB)")
        else:
            logging.warning(f"✗ FP16 training may not fit in A100-40GB ({total_memory_fp16:.2f} GB > {a100_40gb} GB)")
        
        # Recommendations
        logging.info("")
        logging.info("=== Recommendations ===")
        if total_memory_fp16 > a100_40gb:
            logging.info("- Use gradient accumulation with smaller batch sizes")
            logging.info("- Enable mixed precision training")
            logging.info("- Consider gradient checkpointing")
            logging.info("- Reduce sequence length if possible")
        
        return True
        
    except Exception as e:
        logging.error(f"Memory estimation failed: {e}")
        return False

def main():
    """Main function to run memory checks."""
    logging.info("Starting memory requirements check...")
    
    # Check GPU availability
    if not check_gpu_memory():
        logging.error("GPU check failed, exiting")
        return 1
    
    # Estimate memory for different batch sizes
    for batch_size in [1, 2, 4, 8]:
        logging.info(f"\n{'='*50}")
        logging.info(f"Checking batch size: {batch_size}")
        logging.info(f"{'='*50}")
        estimate_model_memory(batch_size)
    
    logging.info("\nMemory check completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())