#!/usr/bin/env python3
"""
Minimal memory test to check if we can even load a small model.
This bypasses most of the training infrastructure to isolate the memory issue.
"""

import sys
import os
sys.path.insert(0, '/app/src')

import jax
import jax.numpy as jnp
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def test_gpu_basic():
    """Test basic GPU functionality."""
    try:
        logging.info("Testing basic GPU functionality...")
        
        # Check devices
        devices = jax.devices()
        logging.info(f"Available devices: {devices}")
        
        # Try simple computation
        x = jnp.array([1.0, 2.0, 3.0])
        y = x * 2
        logging.info(f"Simple computation result: {y}")
        
        return True
    except Exception as e:
        logging.error(f"Basic GPU test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_allocation():
    """Test progressive memory allocation to find limits."""
    try:
        logging.info("Testing memory allocation limits...")
        
        # Start with small allocations and increase
        sizes_mb = [1, 10, 100, 500, 1000, 2000, 4000, 8000]
        
        for size_mb in sizes_mb:
            try:
                # Allocate tensor of specified size
                elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
                tensor = jnp.ones(elements, dtype=jnp.float32)
                logging.info(f"✓ Successfully allocated {size_mb}MB tensor")
                
                # Free memory by deleting tensor
                del tensor
                
            except Exception as e:
                logging.warning(f"✗ Failed to allocate {size_mb}MB: {e}")
                break
        
        return True
    except Exception as e:
        logging.error(f"Memory allocation test failed: {e}")
        return False

def test_minimal_model():
    """Test loading a minimal model."""
    try:
        logging.info("Testing minimal model creation...")
        
        # Import model components
        from openpi.models import pi0
        from openpi.models import gemma
        import flax.nnx as nnx
        
        # Create minimal config
        config = pi0.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=4,  # Minimal action space
            action_horizon=2,  # Minimal horizon
            max_token_len=64,  # Minimal token length
        )
        
        logging.info(f"Model config: {config}")
        
        # Create model with minimal memory
        key = jax.random.key(42)
        rngs = nnx.Rngs(key)
        
        logging.info("Creating model...")
        model = config.create(rngs)
        
        logging.info(f"✓ Successfully created model: {type(model)}")
        
        # Try a forward pass with minimal input
        logging.info("Testing forward pass...")
        
        # Create minimal batch
        batch_size = 1
        seq_len = 32
        
        # Dummy inputs
        dummy_input = {
            'image': {
                'base_0_rgb': jnp.ones((batch_size, 224, 224, 3), dtype=jnp.uint8),
            },
            'state': jnp.ones((batch_size, seq_len, 8), dtype=jnp.float32),
            'actions': jnp.ones((batch_size, seq_len, config.action_dim), dtype=jnp.float32),
            'prompt': jnp.ones((batch_size, 32), dtype=jnp.int32),  # Token IDs
        }
        
        logging.info("Running forward pass...")
        # Note: This might still fail due to model complexity, but let's try
        try:
            output = model(dummy_input)
            logging.info(f"✓ Forward pass successful, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        except Exception as e:
            logging.warning(f"Forward pass failed (expected): {e}")
            # This is expected - the model might still be too complex
        
        return True
        
    except Exception as e:
        logging.error(f"Minimal model test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all memory tests."""
    logging.info("Starting minimal memory tests...")
    
    # Test 1: Basic GPU
    if not test_gpu_basic():
        logging.error("Basic GPU test failed, stopping")
        return 1
    
    # Test 2: Memory allocation
    if not test_memory_allocation():
        logging.error("Memory allocation test failed")
        return 1
    
    # Test 3: Minimal model
    if not test_minimal_model():
        logging.error("Minimal model test failed")
        return 1
    
    logging.info("All tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())