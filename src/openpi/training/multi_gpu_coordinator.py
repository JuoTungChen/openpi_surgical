"""JAX multi-GPU coordination for efficient distributed training."""

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


@dataclasses.dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU coordination."""
    
    # Number of devices to use for FSDP sharding
    fsdp_devices: int = 1
    
    # Minimum parameter size (in MB) to consider for sharding
    min_shard_size_mb: int = 4
    
    # Whether to enable dynamic device count configuration
    enable_dynamic_device_count: bool = True
    
    # Whether to log sharding decisions
    log_sharding_decisions: bool = True
    
    # Memory threshold for automatic sharding adjustments
    memory_threshold: float = 0.9
    
    # Whether to enable load balancing across devices
    enable_load_balancing: bool = True
    
    # Communication backend for gradient synchronization
    communication_backend: str = "nccl"  # or "gloo"


@dataclasses.dataclass
class DeviceInfo:
    """Information about a JAX device."""
    
    device_id: int
    device_kind: str  # "gpu", "tpu", etc.
    memory_gb: float
    utilization: float = 0.0
    is_available: bool = True


class MultiGPUCoordinator:
    """Coordinates JAX multi-GPU training with proper mesh and sharding management."""
    
    def __init__(self, config: MultiGPUConfig):
        self.config = config
        self._mesh: Optional[jax.sharding.Mesh] = None
        self._device_info: List[DeviceInfo] = []
        self._sharding_cache: Dict[str, jax.sharding.NamedSharding] = {}
        self._load_balancing_stats: Dict[int, Dict[str, float]] = {}
        
        # Initialize device information
        self._initialize_devices()
        
    def _initialize_devices(self):
        """Initialize device information and validate configuration."""
        devices = jax.devices()
        self._device_info = []
        
        for i, device in enumerate(devices):
            device_kind = device.device_kind.lower()
            
            # Estimate memory based on device type
            if device_kind == "gpu":
                # Try to get actual memory info, fallback to estimates
                try:
                    # This is device-specific and may not work on all platforms
                    memory_gb = 8.0  # Default assumption
                    if hasattr(device, 'memory_stats'):
                        stats = device.memory_stats()
                        memory_gb = stats.get("bytes_limit", 8 * 1024**3) / (1024**3)
                except Exception:
                    memory_gb = 8.0
            elif device_kind == "tpu":
                memory_gb = 16.0  # TPU v3/v4 typical memory
            else:
                memory_gb = 4.0  # Conservative default
                
            device_info = DeviceInfo(
                device_id=i,
                device_kind=device_kind,
                memory_gb=memory_gb,
                is_available=True
            )
            self._device_info.append(device_info)
            
        logging.info(f"Initialized {len(self._device_info)} devices: {[d.device_kind for d in self._device_info]}")
        
        # Validate FSDP device count
        if self.config.enable_dynamic_device_count:
            self._adjust_fsdp_device_count()
        else:
            self._validate_fsdp_device_count()
    
    def _adjust_fsdp_device_count(self):
        """Dynamically adjust FSDP device count based on available devices."""
        total_devices = len(self._device_info)
        
        if total_devices == 1:
            self.config.fsdp_devices = 1
            logging.info("Single device detected, disabling FSDP sharding")
            return
            
        # Find the largest divisor of total_devices that's <= requested fsdp_devices
        requested_fsdp = self.config.fsdp_devices
        best_fsdp = 1
        
        for candidate in range(1, min(requested_fsdp + 1, total_devices + 1)):
            if total_devices % candidate == 0:
                best_fsdp = candidate
                
        if best_fsdp != requested_fsdp:
            logging.info(f"Adjusted FSDP device count from {requested_fsdp} to {best_fsdp} "
                        f"to match available devices ({total_devices})")
            self.config.fsdp_devices = best_fsdp
    
    def _validate_fsdp_device_count(self):
        """Validate that FSDP device count is compatible with available devices."""
        total_devices = len(self._device_info)
        
        if total_devices % self.config.fsdp_devices != 0:
            raise ValueError(
                f"Number of devices ({total_devices}) must be divisible by "
                f"FSDP devices ({self.config.fsdp_devices})"
            )
    
    def create_mesh(self) -> jax.sharding.Mesh:
        """Create and cache JAX mesh for multi-GPU coordination.
        
        Returns:
            JAX mesh configured for the current device setup
        """
        if self._mesh is not None:
            return self._mesh
            
        total_devices = len(self._device_info)
        fsdp_devices = self.config.fsdp_devices
        
        if total_devices == 1 or fsdp_devices == 1:
            # Single device or no FSDP - create simple mesh
            mesh_shape = (total_devices, 1)
            axis_names = (sharding.BATCH_AXIS, sharding.FSDP_AXIS)
        else:
            # Multi-device FSDP mesh
            batch_devices = total_devices // fsdp_devices
            mesh_shape = (batch_devices, fsdp_devices)
            axis_names = (sharding.BATCH_AXIS, sharding.FSDP_AXIS)
        
        self._mesh = jax.make_mesh(mesh_shape, axis_names)
        
        logging.info(f"Created JAX mesh with shape {mesh_shape} and axes {axis_names}")
        logging.info(f"Mesh devices: {[d.id for d in self._mesh.devices.flat]}")
        
        return self._mesh
    
    def get_mesh(self) -> jax.sharding.Mesh:
        """Get the current mesh, creating it if necessary."""
        if self._mesh is None:
            return self.create_mesh()
        return self._mesh
    
    def setup_fsdp_sharding(
        self, 
        train_state: training_utils.TrainState,
        log_decisions: Optional[bool] = None
    ) -> jax.sharding.NamedSharding:
        """Configure FSDP sharding for model parameters with memory-aware algorithms.
        
        Args:
            train_state: Training state containing model parameters
            log_decisions: Whether to log sharding decisions (uses config default if None)
            
        Returns:
            NamedSharding for the training state
        """
        mesh = self.get_mesh()
        log_decisions = log_decisions if log_decisions is not None else self.config.log_sharding_decisions
        
        # Create cache key for this configuration
        cache_key = f"train_state_fsdp_{mesh.shape}_{self.config.min_shard_size_mb}"
        
        if cache_key in self._sharding_cache:
            logging.debug(f"Using cached sharding for {cache_key}")
            return self._sharding_cache[cache_key]
        
        logging.info("Setting up FSDP sharding for training state...")
        start_time = time.time()
        
        # Apply FSDP sharding with memory-aware configuration
        train_state_sharding = sharding.fsdp_sharding(
            train_state,
            mesh,
            min_size_mbytes=self.config.min_shard_size_mb,
            log=log_decisions,
        )
        
        setup_time = time.time() - start_time
        logging.info(f"FSDP sharding setup completed in {setup_time:.2f}s")
        
        # Cache the sharding for future use
        self._sharding_cache[cache_key] = train_state_sharding
        
        # Log sharding statistics
        if log_decisions:
            self._log_sharding_statistics(train_state, train_state_sharding)
        
        return train_state_sharding
    
    def _log_sharding_statistics(
        self, 
        train_state: training_utils.TrainState,
        train_state_sharding: jax.sharding.NamedSharding
    ):
        """Log detailed sharding statistics."""
        try:
            # Count sharded vs replicated parameters
            sharded_count = 0
            replicated_count = 0
            total_params = 0
            sharded_memory_mb = 0
            total_memory_mb = 0
            
            def analyze_sharding(path, param, shard_spec):
                nonlocal sharded_count, replicated_count, total_params, sharded_memory_mb, total_memory_mb
                
                if hasattr(param, 'shape') and hasattr(param, 'dtype'):
                    param_size_mb = np.prod(param.shape) * np.dtype(param.dtype).itemsize / (1024 * 1024)
                    total_memory_mb += param_size_mb
                    total_params += 1
                    
                    # Check if parameter is sharded (has non-None partition spec)
                    if hasattr(shard_spec, 'spec') and shard_spec.spec is not None:
                        if any(axis is not None for axis in shard_spec.spec):
                            sharded_count += 1
                            sharded_memory_mb += param_size_mb
                        else:
                            replicated_count += 1
                    else:
                        replicated_count += 1
            
            # Analyze parameter sharding
            jax.tree_util.tree_map_with_path(
                analyze_sharding,
                train_state.params,
                train_state_sharding
            )
            
            logging.info(f"Sharding Statistics:")
            logging.info(f"  Total parameters: {total_params}")
            logging.info(f"  Sharded parameters: {sharded_count} ({sharded_count/total_params*100:.1f}%)")
            logging.info(f"  Replicated parameters: {replicated_count} ({replicated_count/total_params*100:.1f}%)")
            logging.info(f"  Total memory: {total_memory_mb:.1f} MB")
            logging.info(f"  Sharded memory: {sharded_memory_mb:.1f} MB ({sharded_memory_mb/total_memory_mb*100:.1f}%)")
            logging.info(f"  Memory per device: {total_memory_mb/len(self._device_info):.1f} MB")
            
        except Exception as e:
            logging.warning(f"Failed to compute sharding statistics: {e}")
    
    def create_data_sharding(self) -> jax.sharding.NamedSharding:
        """Create sharding specification for input data.
        
        Returns:
            NamedSharding for input data batches
        """
        mesh = self.get_mesh()
        
        # Data is sharded across both batch and FSDP axes
        data_sharding = jax.sharding.NamedSharding(
            mesh, 
            jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        
        return data_sharding
    
    def create_replicated_sharding(self) -> jax.sharding.NamedSharding:
        """Create replicated sharding specification.
        
        Returns:
            NamedSharding that replicates data across all devices
        """
        mesh = self.get_mesh()
        
        replicated_sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec()
        )
        
        return replicated_sharding
    
    def balance_workload(
        self, 
        batch: Tuple[Any, Any], 
        target_sharding: jax.sharding.NamedSharding
    ) -> Tuple[Any, Any]:
        """Distribute batch across available devices with load balancing.
        
        Args:
            batch: Input batch to distribute
            target_sharding: Target sharding for the batch
            
        Returns:
            Load-balanced batch distributed across devices
        """
        if not self.config.enable_load_balancing:
            return jax.device_put(batch, target_sharding)
        
        try:
            # Update device utilization stats
            self._update_device_utilization()
            
            # For now, use standard device_put with target sharding
            # In a more advanced implementation, we could:
            # 1. Monitor per-device memory usage
            # 2. Adjust batch distribution based on device performance
            # 3. Implement custom sharding based on device capabilities
            
            distributed_batch = jax.device_put(batch, target_sharding)
            
            # Log load balancing stats periodically
            self._log_load_balancing_stats()
            
            return distributed_batch
            
        except Exception as e:
            logging.warning(f"Load balancing failed, using standard distribution: {e}")
            return jax.device_put(batch, target_sharding)
    
    def _update_device_utilization(self):
        """Update device utilization statistics."""
        try:
            devices = jax.devices()
            current_time = time.time()
            
            for i, device in enumerate(devices):
                if i < len(self._device_info):
                    # Try to get device utilization (this is platform-specific)
                    utilization = 0.0
                    
                    try:
                        # This is a placeholder - actual implementation would depend on
                        # the specific JAX backend and device type
                        if hasattr(device, 'memory_stats'):
                            stats = device.memory_stats()
                            bytes_in_use = stats.get("bytes_in_use", 0)
                            bytes_limit = stats.get("bytes_limit", 1)
                            utilization = bytes_in_use / bytes_limit if bytes_limit > 0 else 0.0
                    except Exception:
                        pass
                    
                    self._device_info[i].utilization = utilization
                    
                    # Update load balancing stats
                    if i not in self._load_balancing_stats:
                        self._load_balancing_stats[i] = {}
                    
                    self._load_balancing_stats[i].update({
                        "utilization": utilization,
                        "last_update": current_time,
                    })
                    
        except Exception as e:
            logging.debug(f"Failed to update device utilization: {e}")
    
    def _log_load_balancing_stats(self):
        """Log load balancing statistics periodically."""
        # Only log every 100 calls to avoid spam
        if not hasattr(self, '_load_balance_call_count'):
            self._load_balance_call_count = 0
        
        self._load_balance_call_count += 1
        
        if self._load_balance_call_count % 100 == 0:
            logging.debug("Load Balancing Stats:")
            for i, device_info in enumerate(self._device_info):
                stats = self._load_balancing_stats.get(i, {})
                logging.debug(f"  Device {i}: {device_info.device_kind}, "
                            f"utilization={device_info.utilization:.2%}, "
                            f"memory={device_info.memory_gb:.1f}GB")
    
    def get_device_count(self) -> int:
        """Get the total number of available devices."""
        return len(self._device_info)
    
    def get_fsdp_device_count(self) -> int:
        """Get the number of devices used for FSDP sharding."""
        return self.config.fsdp_devices
    
    def get_batch_device_count(self) -> int:
        """Get the number of devices used for batch parallelism."""
        return self.get_device_count() // self.get_fsdp_device_count()
    
    def get_device_info(self) -> List[DeviceInfo]:
        """Get information about all devices."""
        return self._device_info.copy()
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get multi-GPU coordination statistics."""
        mesh = self._mesh
        
        stats = {
            "total_devices": len(self._device_info),
            "fsdp_devices": self.config.fsdp_devices,
            "batch_devices": self.get_batch_device_count(),
            "mesh_shape": mesh.shape if mesh else None,
            "mesh_axis_names": mesh.axis_names if mesh else None,
            "sharding_cache_size": len(self._sharding_cache),
            "load_balancing_enabled": self.config.enable_load_balancing,
            "dynamic_device_count": self.config.enable_dynamic_device_count,
        }
        
        # Add device-specific stats
        device_stats = []
        for i, device_info in enumerate(self._device_info):
            device_stat = {
                "device_id": device_info.device_id,
                "device_kind": device_info.device_kind,
                "memory_gb": device_info.memory_gb,
                "utilization": device_info.utilization,
                "is_available": device_info.is_available,
            }
            device_stats.append(device_stat)
        
        stats["devices"] = device_stats
        
        return stats
    
    def validate_sharding_compatibility(
        self, 
        train_state: training_utils.TrainState
    ) -> Tuple[bool, List[str]]:
        """Validate that the training state is compatible with current sharding setup.
        
        Args:
            train_state: Training state to validate
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []
        
        try:
            mesh = self.get_mesh()
            
            # Check if mesh is properly configured
            if mesh is None:
                issues.append("No mesh configured")
                return False, issues
            
            # Check device count compatibility
            expected_devices = np.prod(mesh.shape)
            actual_devices = len(self._device_info)
            
            if expected_devices != actual_devices:
                issues.append(f"Mesh expects {expected_devices} devices but {actual_devices} available")
            
            # Check FSDP compatibility
            if self.config.fsdp_devices > 1:
                if actual_devices % self.config.fsdp_devices != 0:
                    issues.append(f"Device count {actual_devices} not divisible by FSDP devices {self.config.fsdp_devices}")
            
            # Try to create sharding to validate compatibility
            try:
                _ = self.setup_fsdp_sharding(train_state, log_decisions=False)
            except Exception as e:
                issues.append(f"Failed to create FSDP sharding: {e}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation failed: {e}")
            return False, issues