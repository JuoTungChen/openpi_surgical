"""Tests for multi-GPU coordinator functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training.multi_gpu_coordinator import MultiGPUCoordinator, MultiGPUConfig, DeviceInfo
from openpi.training.gradient_synchronization import GradientSyncConfig, AsyncGradientSynchronizer
from openpi.training.utils import TrainState
from flax import nnx


def test_multi_gpu_config():
    """Test MultiGPUConfig initialization and validation."""
    config = MultiGPUConfig(
        fsdp_devices=2,
        min_shard_size_mb=8,
        enable_dynamic_device_count=True,
        log_sharding_decisions=False,
    )
    
    assert config.fsdp_devices == 2
    assert config.min_shard_size_mb == 8
    assert config.enable_dynamic_device_count is True
    assert config.log_sharding_decisions is False


def test_device_info():
    """Test DeviceInfo dataclass."""
    device_info = DeviceInfo(
        device_id=0,
        device_kind="gpu",
        memory_gb=8.0,
        utilization=0.5,
        is_available=True,
    )
    
    assert device_info.device_id == 0
    assert device_info.device_kind == "gpu"
    assert device_info.memory_gb == 8.0
    assert device_info.utilization == 0.5
    assert device_info.is_available is True


def test_multi_gpu_coordinator_initialization():
    """Test MultiGPUCoordinator initialization."""
    config = MultiGPUConfig(
        fsdp_devices=1,  # Use 1 to avoid device count issues in testing
        enable_dynamic_device_count=True,
        log_sharding_decisions=False,
    )
    
    coordinator = MultiGPUCoordinator(config)
    
    assert coordinator.config == config
    assert len(coordinator._device_info) > 0
    assert coordinator._mesh is None  # Not created yet


def test_mesh_creation():
    """Test JAX mesh creation."""
    config = MultiGPUConfig(
        fsdp_devices=1,
        enable_dynamic_device_count=True,
        log_sharding_decisions=False,
    )
    
    coordinator = MultiGPUCoordinator(config)
    mesh = coordinator.create_mesh()
    
    assert mesh is not None
    assert coordinator._mesh is mesh  # Should be cached
    
    # Test mesh properties
    assert len(mesh.shape) == 2
    assert mesh.axis_names == ("batch", "fsdp")


def test_device_count_adjustment():
    """Test dynamic device count adjustment."""
    total_devices = jax.device_count()
    
    config = MultiGPUConfig(
        fsdp_devices=total_devices + 1,  # Request more than available
        enable_dynamic_device_count=True,
        log_sharding_decisions=False,
    )
    
    coordinator = MultiGPUCoordinator(config)
    
    # Should adjust to valid value
    assert coordinator.config.fsdp_devices <= total_devices
    assert total_devices % coordinator.config.fsdp_devices == 0


def test_sharding_creation():
    """Test sharding specification creation."""
    config = MultiGPUConfig(
        fsdp_devices=1,
        log_sharding_decisions=False,
    )
    
    coordinator = MultiGPUCoordinator(config)
    
    # Test data sharding
    data_sharding = coordinator.create_data_sharding()
    assert data_sharding is not None
    
    # Test replicated sharding
    replicated_sharding = coordinator.create_replicated_sharding()
    assert replicated_sharding is not None


def test_coordination_stats():
    """Test coordination statistics collection."""
    config = MultiGPUConfig(
        fsdp_devices=1,
        log_sharding_decisions=False,
    )
    
    coordinator = MultiGPUCoordinator(config)
    mesh = coordinator.create_mesh()
    
    stats = coordinator.get_coordination_stats()
    
    assert "total_devices" in stats
    assert "fsdp_devices" in stats
    assert "batch_devices" in stats
    assert "mesh_shape" in stats
    assert "devices" in stats
    
    assert stats["total_devices"] > 0
    assert stats["fsdp_devices"] == config.fsdp_devices
    assert stats["mesh_shape"] == mesh.shape


def test_gradient_sync_config():
    """Test GradientSyncConfig initialization."""
    config = GradientSyncConfig(
        sync_strategy="allreduce",
        enable_compression=True,
        compression_method="quantization",
        enable_async_comm=True,
    )
    
    assert config.sync_strategy == "allreduce"
    assert config.enable_compression is True
    assert config.compression_method == "quantization"
    assert config.enable_async_comm is True


def test_gradient_synchronizer_initialization():
    """Test AsyncGradientSynchronizer initialization."""
    config = GradientSyncConfig(
        sync_strategy="allreduce",
        log_sync_stats=False,
    )
    
    # Create a simple mesh for testing
    mesh = jax.make_mesh((1, 1), ("batch", "fsdp"))
    
    synchronizer = AsyncGradientSynchronizer(config, mesh)
    
    assert synchronizer.config == config
    assert synchronizer.mesh == mesh
    assert synchronizer.compressor is not None


def test_gradient_compression():
    """Test gradient compression functionality."""
    config = GradientSyncConfig(
        enable_compression=True,
        compression_method="quantization",
    )
    
    mesh = jax.make_mesh((1, 1), ("batch", "fsdp"))
    synchronizer = AsyncGradientSynchronizer(config, mesh)
    
    # Create test gradients
    test_gradients = {
        "layer1": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        "layer2": jnp.array([0.5, -0.5], dtype=jnp.float32),
    }
    
    # Test compression
    compressed_grads, metadata = synchronizer.compressor.compress_gradients(test_gradients)
    
    assert compressed_grads is not None
    assert metadata is not None
    assert "method" in metadata
    
    # Test decompression
    decompressed_grads = synchronizer.compressor.decompress_gradients(compressed_grads, metadata)
    
    assert decompressed_grads is not None
    assert len(decompressed_grads) == len(test_gradients)


def test_load_balancing():
    """Test load balancing functionality."""
    config = MultiGPUConfig(
        fsdp_devices=1,
        enable_load_balancing=True,
        log_sharding_decisions=False,
    )
    
    coordinator = MultiGPUCoordinator(config)
    data_sharding = coordinator.create_data_sharding()
    
    # Create test batch (simplified)
    test_batch = (
        {"images": {"cam1": jnp.ones((2, 64, 64, 3))}},  # observation
        jnp.ones((2, 10)),  # actions
    )
    
    # Test load balancing
    balanced_batch = coordinator.balance_workload(test_batch, data_sharding)
    
    assert balanced_batch is not None
    assert len(balanced_batch) == 2  # observation, actions


if __name__ == "__main__":
    # Run basic tests
    test_multi_gpu_config()
    test_device_info()
    test_multi_gpu_coordinator_initialization()
    test_mesh_creation()
    test_device_count_adjustment()
    test_sharding_creation()
    test_coordination_stats()
    test_gradient_sync_config()
    test_gradient_synchronizer_initialization()
    test_gradient_compression()
    test_load_balancing()
    
    print("All multi-GPU coordinator tests passed!")