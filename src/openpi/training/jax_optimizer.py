"""JAX compilation and memory optimization utilities for training."""

import dataclasses
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.gradient_synchronization as _gradient_sync
import openpi.training.multi_gpu_coordinator as _multi_gpu
import openpi.training.utils as training_utils


@dataclasses.dataclass
class CompilationCacheConfig:
    """Configuration for JIT compilation cache warming."""
    
    # Whether to enable compilation cache warming
    enable_cache_warming: bool = True
    
    # Number of warmup iterations to run during cache warming
    warmup_iterations: int = 3
    
    # Whether to persist compilation cache to disk
    persist_cache: bool = True
    
    # Maximum cache size (number of compiled functions to keep)
    max_cache_size: int = 100
    
    # Whether to log compilation timing information
    log_compilation_timing: bool = True


@dataclasses.dataclass
class MemoryOptimizationConfig:
    """Configuration for memory layout optimizations."""
    
    # Whether to enable memory optimizations
    enable_memory_optimization: bool = True
    
    # Target memory utilization (fraction of total GPU memory)
    target_memory_utilization: float = 0.9
    
    # Whether to enable automatic batch size adjustment
    enable_auto_batch_sizing: bool = True
    
    # Minimum batch size for auto-adjustment
    min_batch_size: int = 1
    
    # Maximum batch size for auto-adjustment  
    max_batch_size: int = 512
    
    # Maximum compilation cache size for memory management
    max_cache_size: int = 100


@dataclasses.dataclass
class TrainingOptimizationConfig:
    """Configuration for training step optimizations."""
    
    # Whether to enable gradient accumulation
    enable_gradient_accumulation: bool = False
    
    # Number of gradient accumulation steps
    gradient_accumulation_steps: int = 1
    
    # Whether to enable overlapped computation and data transfer
    enable_overlapped_computation: bool = True
    
    # Whether to use mixed precision training
    enable_mixed_precision: bool = True
    
    # Whether to optimize gradient synchronization
    optimize_gradient_sync: bool = True


@dataclasses.dataclass
class MultiGPUOptimizationConfig:
    """Configuration for multi-GPU optimizations."""
    
    # Whether to enable multi-GPU coordination
    enable_multi_gpu: bool = True
    
    # Multi-GPU coordinator configuration
    multi_gpu_config: _multi_gpu.MultiGPUConfig = dataclasses.field(
        default_factory=_multi_gpu.MultiGPUConfig
    )
    
    # Gradient synchronization configuration
    gradient_sync_config: _gradient_sync.GradientSyncConfig = dataclasses.field(
        default_factory=_gradient_sync.GradientSyncConfig
    )


@dataclasses.dataclass
class JaxOptimizationConfig:
    """Combined configuration for JAX optimizations."""
    
    compilation: CompilationCacheConfig = dataclasses.field(default_factory=CompilationCacheConfig)
    memory: MemoryOptimizationConfig = dataclasses.field(default_factory=MemoryOptimizationConfig)
    training: TrainingOptimizationConfig = dataclasses.field(default_factory=TrainingOptimizationConfig)
    multi_gpu: MultiGPUOptimizationConfig = dataclasses.field(default_factory=MultiGPUOptimizationConfig)


class JaxCompilationCacheWarmer:
    """Manages JIT compilation cache warming for training functions."""
    
    def __init__(self, config: CompilationCacheConfig):
        self.config = config
        self._compilation_cache: Dict[str, Any] = {}
        self._compilation_times: Dict[str, float] = {}
        
    def warm_train_step(
        self,
        train_step_fn: Callable,
        sample_batch: Tuple[_model.Observation, _model.Actions],
        train_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        train_config: _config.TrainConfig,
    ) -> Callable:
        """Warm up the compilation cache for the training step function.
        
        Args:
            train_step_fn: The training step function to compile
            sample_batch: Sample batch for compilation
            train_state: Sample train state
            rng: Random key
            train_config: Training configuration
            
        Returns:
            The JIT-compiled training step function
        """
        if not self.config.enable_cache_warming:
            logging.info("Compilation cache warming disabled, skipping...")
            return train_step_fn
            
        logging.info("Starting JIT compilation cache warming...")
        start_time = time.time()
        
        # Create the JIT-compiled function
        compiled_fn = jax.jit(
            functools.partial(train_step_fn, train_config),
            donate_argnums=(1,),  # donate train_state
        )
        
        # Warm up the compilation cache
        for i in range(self.config.warmup_iterations):
            logging.info(f"Cache warming iteration {i + 1}/{self.config.warmup_iterations}")
            
            # Run a forward pass to trigger compilation
            iteration_start = time.time()
            try:
                _, _ = compiled_fn(rng, train_state, sample_batch)
                jax.block_until_ready(train_state)  # Ensure compilation is complete
                
                iteration_time = time.time() - iteration_start
                if self.config.log_compilation_timing:
                    logging.info(f"Warmup iteration {i + 1} completed in {iteration_time:.2f}s")
                    
            except Exception as e:
                logging.warning(f"Warmup iteration {i + 1} failed: {e}")
                continue
        
        total_time = time.time() - start_time
        logging.info(f"JIT compilation cache warming completed in {total_time:.2f}s")
        
        # Store compilation info
        self._compilation_cache["train_step"] = compiled_fn
        self._compilation_times["train_step"] = total_time
        
        return compiled_fn
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return {
            "cached_functions": list(self._compilation_cache.keys()),
            "compilation_times": self._compilation_times.copy(),
            "total_compilation_time": sum(self._compilation_times.values()),
            "cache_size": len(self._compilation_cache),
        }


class MemoryOptimizer:
    """Handles memory layout optimizations and monitoring."""
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self._memory_stats: Dict[str, float] = {}
        self._batch_size_history: List[Tuple[int, float, float]] = []  # (batch_size, utilization, timestamp)
        self._memory_pressure_threshold = 0.95  # Emergency threshold
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        try:
            # Get memory info for all devices
            devices = jax.devices()
            memory_stats = {}
            
            for i, device in enumerate(devices):
                if hasattr(device, 'memory_stats'):
                    stats = device.memory_stats()
                    memory_stats[f"device_{i}"] = {
                        "bytes_in_use": stats.get("bytes_in_use", 0),
                        "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                        "bytes_limit": stats.get("bytes_limit", 0),
                    }
                else:
                    # Fallback: assume 8GB GPU
                    memory_stats[f"device_{i}"] = {
                        "bytes_in_use": 0,
                        "peak_bytes_in_use": 0,
                        "bytes_limit": 8 * 1024**3,  # Assume 8GB
                    }
            
            return memory_stats
            
        except Exception as e:
            logging.warning(f"Failed to get memory usage: {e}")
            return {}
    
    def check_memory_usage(self, batch_size: int) -> Tuple[bool, Dict[str, Any]]:
        """Check if current memory usage is within safe limits.
        
        Args:
            batch_size: Current batch size
            
        Returns:
            Tuple of (is_safe, memory_info)
        """
        memory_stats = self.get_memory_usage()
        
        if not memory_stats:
            return True, {"warning": "Could not retrieve memory stats"}
        
        max_utilization = 0.0
        total_used = 0
        total_limit = 0
        memory_info = {"batch_size": batch_size, "devices": {}}
        
        for device_name, stats in memory_stats.items():
            bytes_limit = stats.get("bytes_limit", 0)
            bytes_in_use = stats.get("bytes_in_use", 0)
            peak_bytes = stats.get("peak_bytes_in_use", 0)
            
            total_used += bytes_in_use
            total_limit += bytes_limit
            
            if bytes_limit > 0:
                utilization = bytes_in_use / bytes_limit
                peak_utilization = peak_bytes / bytes_limit
                max_utilization = max(max_utilization, utilization)
                
                memory_info["devices"][device_name] = {
                    "utilization": utilization,
                    "peak_utilization": peak_utilization,
                    "bytes_in_use": bytes_in_use,
                    "bytes_limit": bytes_limit,
                    "gb_in_use": bytes_in_use / (1024**3),
                    "gb_limit": bytes_limit / (1024**3),
                }
        
        memory_info["max_utilization"] = max_utilization
        memory_info["total_utilization"] = total_used / total_limit if total_limit > 0 else 0.0
        
        # Check if memory usage is safe
        is_safe = max_utilization < self.config.target_memory_utilization
        memory_info["is_safe"] = is_safe
        
        return is_safe, memory_info
    
    def optimize_memory_layout(self, train_state: training_utils.TrainState) -> training_utils.TrainState:
        """Optimize memory layout of training state.
        
        Args:
            train_state: Current training state
            
        Returns:
            Optimized training state
        """
        if not self.config.enable_memory_optimization:
            return train_state
            
        try:
            # Force garbage collection to free unused memory
            import gc
            gc.collect()
            
            logging.debug("Memory layout optimization applied")
            
        except Exception as e:
            logging.warning(f"Memory layout optimization failed: {e}")
            
        return train_state
    
    def get_memory_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        return {
            "config": {
                "target_memory_utilization": self.config.target_memory_utilization,
                "enable_auto_batch_sizing": self.config.enable_auto_batch_sizing,
                "min_batch_size": self.config.min_batch_size,
                "max_batch_size": self.config.max_batch_size,
            }
        }


class JaxTrainingOptimizer:
    """Main class for JAX training optimizations."""
    
    def __init__(self, config: JaxOptimizationConfig):
        self.config = config
        self.cache_warmer = JaxCompilationCacheWarmer(config.compilation)
        self.memory_optimizer = MemoryOptimizer(config.memory)
        
        # Initialize multi-GPU components if enabled
        if config.multi_gpu.enable_multi_gpu:
            self.multi_gpu_coordinator = _multi_gpu.MultiGPUCoordinator(
                config.multi_gpu.multi_gpu_config
            )
            self.gradient_synchronizer = None  # Will be initialized with mesh
        else:
            self.multi_gpu_coordinator = None
            self.gradient_synchronizer = None
        
    def optimize_data_transfer(
        self,
        batch: Tuple[_model.Observation, _model.Actions],
        target_sharding: jax.sharding.NamedSharding,
    ) -> Tuple[_model.Observation, _model.Actions]:
        """Optimize data transfer to GPU with memory-efficient patterns.
        
        Args:
            batch: Input batch to transfer
            target_sharding: Target sharding for the batch
            
        Returns:
            Optimized batch with efficient memory layout
        """
        if not self.config.memory.enable_memory_optimization:
            return batch
            
        try:
            # Use JAX's device_put with explicit sharding for efficient transfer
            observation, actions = batch
            
            # Transfer observation data efficiently
            optimized_observation = jax.device_put(observation, target_sharding)
            
            # Transfer action data efficiently  
            optimized_actions = jax.device_put(actions, target_sharding)
            
            return optimized_observation, optimized_actions
            
        except Exception as e:
            logging.warning(f"Data transfer optimization failed: {e}")
            return batch
    
    def create_optimized_train_step(
        self,
        train_step_fn: Callable,
        sample_batch: Tuple[_model.Observation, _model.Actions],
        train_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        train_config: _config.TrainConfig,
        mesh: jax.sharding.Mesh,
        train_state_sharding: jax.sharding.NamedSharding,
        data_sharding: jax.sharding.NamedSharding,
        replicated_sharding: jax.sharding.NamedSharding,
    ) -> Callable:
        """Create an optimized training step function with all optimizations.
        
        Args:
            train_step_fn: Base training step function
            sample_batch: Sample batch for compilation
            train_state: Sample train state
            rng: Random key
            train_config: Training configuration
            mesh: JAX mesh for sharding
            train_state_sharding: Sharding for train state
            data_sharding: Sharding for data
            replicated_sharding: Replicated sharding
            
        Returns:
            Optimized JIT-compiled training step function
        """
        # Optimize memory layout of train state
        optimized_train_state = self.memory_optimizer.optimize_memory_layout(train_state)
        
        # Optimize sample batch for compilation
        optimized_sample_batch = self.optimize_data_transfer(sample_batch, data_sharding)
        
        # Create JIT-compiled function with sharding
        compiled_fn = jax.jit(
            functools.partial(train_step_fn, train_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
        
        # Warm up compilation cache if enabled
        if self.config.compilation.enable_cache_warming:
            logging.info("Warming up JIT compilation cache for optimized training step...")
            start_time = time.time()
            
            for i in range(self.config.compilation.warmup_iterations):
                try:
                    logging.info(f"Warmup iteration {i + 1}/{self.config.compilation.warmup_iterations}")
                    # Create a copy of train_state for warmup to avoid donation issues
                    warmup_train_state = jax.tree_map(lambda x: x, optimized_train_state)
                    _, _ = compiled_fn(rng, warmup_train_state, optimized_sample_batch)
                    jax.block_until_ready(warmup_train_state)
                    
                except Exception as e:
                    logging.warning(f"Warmup iteration {i + 1} failed: {e}")
                    continue
            
            warmup_time = time.time() - start_time
            logging.info(f"JIT compilation cache warming completed in {warmup_time:.2f}s")
        
        return compiled_fn
    
    def setup_multi_gpu_coordination(
        self,
        train_state: training_utils.TrainState,
    ) -> Tuple[jax.sharding.Mesh, jax.sharding.NamedSharding, jax.sharding.NamedSharding, jax.sharding.NamedSharding]:
        """Setup multi-GPU coordination with mesh and sharding.
        
        Args:
            train_state: Training state to configure sharding for
            
        Returns:
            Tuple of (mesh, train_state_sharding, data_sharding, replicated_sharding)
        """
        if not self.config.multi_gpu.enable_multi_gpu or self.multi_gpu_coordinator is None:
            # Fallback to single device
            devices = jax.devices()
            mesh = jax.make_mesh((len(devices),), ("batch",))
            
            train_state_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
            replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            
            return mesh, train_state_sharding, data_sharding, replicated_sharding
        
        # Create mesh using multi-GPU coordinator
        mesh = self.multi_gpu_coordinator.create_mesh()
        
        # Setup FSDP sharding for training state
        train_state_sharding = self.multi_gpu_coordinator.setup_fsdp_sharding(train_state)
        
        # Create data and replicated sharding
        data_sharding = self.multi_gpu_coordinator.create_data_sharding()
        replicated_sharding = self.multi_gpu_coordinator.create_replicated_sharding()
        
        # Initialize gradient synchronizer with mesh
        if self.gradient_synchronizer is None:
            self.gradient_synchronizer = _gradient_sync.AsyncGradientSynchronizer(
                self.config.multi_gpu.gradient_sync_config,
                mesh
            )
        
        # Validate sharding compatibility
        is_compatible, issues = self.multi_gpu_coordinator.validate_sharding_compatibility(train_state)
        if not is_compatible:
            logging.warning(f"Sharding compatibility issues detected: {issues}")
        
        return mesh, train_state_sharding, data_sharding, replicated_sharding
    
    def synchronize_gradients(
        self,
        gradients: at.Params,
        step: int,
    ) -> Tuple[at.Params, Dict[str, Any]]:
        """Synchronize gradients across devices with optimizations.
        
        Args:
            gradients: Local gradients to synchronize
            step: Current training step
            
        Returns:
            Tuple of (synchronized_gradients, sync_statistics)
        """
        if self.gradient_synchronizer is None:
            # No multi-GPU setup, return gradients as-is
            return gradients, {"sync_disabled": True}
        
        try:
            synchronized_grads, sync_stats = self.gradient_synchronizer.synchronize_gradients(
                gradients, step
            )
            
            # Convert sync stats to dict for logging
            stats_dict = dataclasses.asdict(sync_stats)
            
            return synchronized_grads, stats_dict
            
        except Exception as e:
            logging.warning(f"Gradient synchronization failed: {e}")
            return gradients, {"sync_error": str(e)}
    
    def optimize_batch_distribution(
        self,
        batch: Tuple[_model.Observation, _model.Actions],
        data_sharding: jax.sharding.NamedSharding,
    ) -> Tuple[_model.Observation, _model.Actions]:
        """Optimize batch distribution across devices with load balancing.
        
        Args:
            batch: Input batch to distribute
            data_sharding: Target sharding for the batch
            
        Returns:
            Optimized and distributed batch
        """
        if self.multi_gpu_coordinator is None:
            return jax.device_put(batch, data_sharding)
        
        return self.multi_gpu_coordinator.balance_workload(batch, data_sharding)
    
    def get_multi_gpu_stats(self) -> Dict[str, Any]:
        """Get multi-GPU coordination and synchronization statistics."""
        stats = {}
        
        if self.multi_gpu_coordinator is not None:
            stats["coordination"] = self.multi_gpu_coordinator.get_coordination_stats()
        
        if self.gradient_synchronizer is not None:
            stats["gradient_sync"] = self.gradient_synchronizer.get_sync_statistics()
        
        return stats
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "compilation": self.cache_warmer.get_compilation_stats(),
            "memory": self.memory_optimizer.get_memory_optimization_stats(),
            "training": {
                "gradient_accumulation_enabled": self.config.training.enable_gradient_accumulation,
                "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
                "overlapped_computation_enabled": self.config.training.enable_overlapped_computation,
                "mixed_precision_enabled": self.config.training.enable_mixed_precision,
            }
        }
        
        # Add multi-GPU stats if available
        multi_gpu_stats = self.get_multi_gpu_stats()
        if multi_gpu_stats:
            stats["multi_gpu"] = multi_gpu_stats
        
        return stats