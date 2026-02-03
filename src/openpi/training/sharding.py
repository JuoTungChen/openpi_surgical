import contextlib
import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


@dataclasses.dataclass
class ShardingConfig:
    """Configuration for FSDP sharding strategies."""
    
    # Minimum size in MB for sharding consideration
    min_size_mbytes: int = 4
    
    # Maximum size in MB before forcing sharding
    max_size_mbytes: int = 1024
    
    # Memory utilization threshold for dynamic sharding
    memory_threshold: float = 0.8
    
    # Whether to enable memory-aware sharding
    enable_memory_aware_sharding: bool = True
    
    # Whether to enable dynamic sharding based on model size
    enable_dynamic_sharding: bool = True
    
    # Whether to log sharding decisions
    log_sharding_decisions: bool = True
    
    # Performance monitoring interval (in seconds)
    monitoring_interval: float = 10.0


@dataclasses.dataclass
class ShardingStats:
    """Statistics about sharding decisions and performance."""
    
    total_parameters: int = 0
    sharded_parameters: int = 0
    replicated_parameters: int = 0
    total_memory_mb: float = 0.0
    sharded_memory_mb: float = 0.0
    sharding_time_seconds: float = 0.0
    memory_efficiency: float = 0.0
    communication_overhead: float = 0.0


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)


def enhanced_fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    config: ShardingConfig,
    available_memory_gb: Optional[float] = None,
) -> Tuple[Any, ShardingStats]:
    """Enhanced FSDP sharding with memory-aware algorithms and performance monitoring.
    
    Args:
        pytree: A pytree to apply sharding to
        mesh: The mesh being used for sharding
        config: Sharding configuration
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
        
    Returns:
        Tuple of (sharded_pytree, sharding_stats)
    """
    start_time = time.time()
    
    # Initialize statistics
    stats = ShardingStats()
    
    # Auto-detect available memory if not provided
    if available_memory_gb is None:
        available_memory_gb = _estimate_available_memory()
    
    min_size_bytes = config.min_size_mbytes * 2**20
    max_size_bytes = config.max_size_mbytes * 2**20
    
    # Track sharding decisions for statistics
    sharding_decisions = []
    
    def _enhanced_shard_arr(kp, array: jax.ShapeDtypeStruct):
        nonlocal stats
        
        # Count total parameters
        stats.total_parameters += 1
        
        # Skip non-arrays
        if not hasattr(array, "shape"):
            stats.replicated_parameters += 1
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # Calculate array size
        arr_size_bytes = np.prod(array.shape) * np.dtype(array.dtype).itemsize
        arr_size_mb = arr_size_bytes / (2**20)
        stats.total_memory_mb += arr_size_mb
        
        # If FSDP is not used, replicate everything
        if mesh.shape[FSDP_AXIS] == 1:
            stats.replicated_parameters += 1
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # Replicate small arrays (below minimum threshold)
        if arr_size_bytes < min_size_bytes:
            stats.replicated_parameters += 1
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # Replicate scalar and vector arrays
        if len(array.shape) < 2:
            stats.replicated_parameters += 1
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # Memory-aware sharding decision
        if config.enable_memory_aware_sharding:
            memory_pressure = stats.total_memory_mb / (available_memory_gb * 1024)
            
            # Force sharding for large arrays or under memory pressure
            force_shard = (
                arr_size_bytes > max_size_bytes or 
                memory_pressure > config.memory_threshold
            )
        else:
            force_shard = False
        
        # Find the best sharding strategy
        sharding_spec = _find_optimal_sharding(
            array, mesh, force_shard, config.enable_dynamic_sharding
        )
        
        if sharding_spec is not None:
            # Parameter will be sharded
            stats.sharded_parameters += 1
            stats.sharded_memory_mb += arr_size_mb
            
            if config.log_sharding_decisions:
                sharding_info = {
                    "path": jax.tree_util.keystr(kp),
                    "shape": array.shape,
                    "size_mb": arr_size_mb,
                    "sharding_axis": _get_sharding_axis(sharding_spec),
                    "forced": force_shard,
                }
                sharding_decisions.append(sharding_info)
                
                logging.info(
                    f"Sharding {sharding_info['path']} of shape {sharding_info['shape']} "
                    f"({sharding_info['size_mb']:.2f} MB) along axis {sharding_info['sharding_axis']}"
                    f"{' (forced)' if force_shard else ''}"
                )
            
            return jax.sharding.NamedSharding(mesh, sharding_spec)
        else:
            # Parameter will be replicated
            stats.replicated_parameters += 1
            
            if config.log_sharding_decisions:
                logging.warning(
                    f"Could not find valid sharding for {jax.tree_util.keystr(kp)} "
                    f"of shape {array.shape} with mesh shape {mesh.shape}"
                )
            
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Apply sharding
    sharded_pytree = jax.tree_util.tree_map_with_path(_enhanced_shard_arr, pytree)
    
    # Finalize statistics
    stats.sharding_time_seconds = time.time() - start_time
    
    if stats.total_memory_mb > 0:
        stats.memory_efficiency = stats.sharded_memory_mb / stats.total_memory_mb
    
    # Estimate communication overhead (simplified model)
    if stats.sharded_parameters > 0:
        # Rough estimate: communication overhead increases with number of sharded parameters
        # and decreases with mesh size
        base_overhead = 0.1  # 10% base overhead
        sharding_factor = stats.sharded_parameters / stats.total_parameters
        mesh_efficiency = 1.0 / mesh.shape[FSDP_AXIS]
        stats.communication_overhead = base_overhead * sharding_factor * mesh_efficiency
    
    # Log summary statistics
    if config.log_sharding_decisions:
        _log_sharding_summary(stats, mesh, sharding_decisions)
    
    return sharded_pytree, stats


def _estimate_available_memory() -> float:
    """Estimate available GPU memory in GB."""
    try:
        devices = jax.devices()
        if not devices:
            return 8.0  # Default fallback
        
        # Try to get memory info from the first GPU device
        device = devices[0]
        
        if hasattr(device, 'memory_stats'):
            stats = device.memory_stats()
            memory_limit = stats.get("bytes_limit", 8 * 1024**3)
            return memory_limit / (1024**3)
        else:
            # Fallback based on device type
            device_kind = device.device_kind.lower()
            if device_kind == "gpu":
                return 8.0  # Assume 8GB GPU
            elif device_kind == "tpu":
                return 16.0  # Assume 16GB TPU
            else:
                return 4.0  # Conservative default
                
    except Exception as e:
        logging.debug(f"Failed to estimate memory: {e}")
        return 8.0  # Safe default


def _find_optimal_sharding(
    array: jax.ShapeDtypeStruct,
    mesh: jax.sharding.Mesh,
    force_shard: bool,
    enable_dynamic: bool,
) -> Optional[jax.sharding.PartitionSpec]:
    """Find the optimal sharding strategy for an array.
    
    Args:
        array: Array to shard
        mesh: JAX mesh
        force_shard: Whether to force sharding even if not optimal
        enable_dynamic: Whether to use dynamic sharding strategies
        
    Returns:
        PartitionSpec for sharding, or None if should be replicated
    """
    fsdp_size = mesh.shape[FSDP_AXIS]
    
    if enable_dynamic:
        # Dynamic sharding: consider multiple strategies
        strategies = _generate_sharding_strategies(array.shape, fsdp_size)
        best_strategy = _select_best_strategy(strategies, array.shape, fsdp_size)
        
        if best_strategy is not None or force_shard:
            return best_strategy
    else:
        # Original strategy: shard along largest divisible axis
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(array.shape)
        
        for i in axes:
            if array.shape[i] % fsdp_size == 0:
                spec[i] = FSDP_AXIS
                return jax.sharding.PartitionSpec(*spec)
    
    return None


def _generate_sharding_strategies(
    shape: Tuple[int, ...], 
    fsdp_size: int
) -> List[jax.sharding.PartitionSpec]:
    """Generate possible sharding strategies for a given shape."""
    strategies = []
    
    # Strategy 1: Shard along largest divisible axis (original)
    axes = np.argsort(shape)[::-1]
    for i in axes:
        if shape[i] % fsdp_size == 0:
            spec = [None] * len(shape)
            spec[i] = FSDP_AXIS
            strategies.append(jax.sharding.PartitionSpec(*spec))
            break
    
    # Strategy 2: Shard along first axis if divisible (common for weight matrices)
    if len(shape) >= 2 and shape[0] % fsdp_size == 0:
        spec = [FSDP_AXIS] + [None] * (len(shape) - 1)
        strategies.append(jax.sharding.PartitionSpec(*spec))
    
    # Strategy 3: Shard along last axis if divisible (common for embeddings)
    if len(shape) >= 2 and shape[-1] % fsdp_size == 0:
        spec = [None] * (len(shape) - 1) + [FSDP_AXIS]
        strategies.append(jax.sharding.PartitionSpec(*spec))
    
    return strategies


def _select_best_strategy(
    strategies: List[jax.sharding.PartitionSpec],
    shape: Tuple[int, ...],
    fsdp_size: int,
) -> Optional[jax.sharding.PartitionSpec]:
    """Select the best sharding strategy based on heuristics."""
    if not strategies:
        return None
    
    # Score each strategy
    best_strategy = None
    best_score = -1
    
    for strategy in strategies:
        score = _score_sharding_strategy(strategy, shape, fsdp_size)
        if score > best_score:
            best_score = score
            best_strategy = strategy
    
    return best_strategy


def _score_sharding_strategy(
    strategy: jax.sharding.PartitionSpec,
    shape: Tuple[int, ...],
    fsdp_size: int,
) -> float:
    """Score a sharding strategy (higher is better)."""
    score = 0.0
    
    # Find the sharded axis
    sharded_axis = None
    for i, spec in enumerate(strategy.spec):
        if spec == FSDP_AXIS:
            sharded_axis = i
            break
    
    if sharded_axis is None:
        return 0.0
    
    # Prefer sharding larger dimensions
    axis_size = shape[sharded_axis]
    score += np.log(axis_size) * 0.5
    
    # Prefer even division
    if axis_size % fsdp_size == 0:
        score += 2.0
        # Bonus for powers of 2
        if (axis_size // fsdp_size) & ((axis_size // fsdp_size) - 1) == 0:
            score += 0.5
    
    # Prefer sharding the first axis for weight matrices (common pattern)
    if sharded_axis == 0 and len(shape) == 2:
        score += 1.0
    
    # Prefer sharding the last axis for embeddings
    if sharded_axis == len(shape) - 1 and len(shape) >= 2:
        score += 0.5
    
    return score


def _get_sharding_axis(spec: jax.sharding.PartitionSpec) -> Optional[int]:
    """Get the axis being sharded in a PartitionSpec."""
    for i, axis_spec in enumerate(spec.spec):
        if axis_spec == FSDP_AXIS:
            return i
    return None


def _log_sharding_summary(
    stats: ShardingStats,
    mesh: jax.sharding.Mesh,
    decisions: List[Dict[str, Any]],
):
    """Log a summary of sharding decisions and statistics."""
    logging.info("=== FSDP Sharding Summary ===")
    logging.info(f"Mesh shape: {mesh.shape}")
    logging.info(f"Total parameters: {stats.total_parameters}")
    logging.info(f"Sharded parameters: {stats.sharded_parameters} ({stats.sharded_parameters/stats.total_parameters*100:.1f}%)")
    logging.info(f"Replicated parameters: {stats.replicated_parameters} ({stats.replicated_parameters/stats.total_parameters*100:.1f}%)")
    logging.info(f"Total memory: {stats.total_memory_mb:.1f} MB")
    logging.info(f"Sharded memory: {stats.sharded_memory_mb:.1f} MB ({stats.memory_efficiency*100:.1f}%)")
    logging.info(f"Memory per device: {stats.total_memory_mb/np.prod(mesh.shape):.1f} MB")
    logging.info(f"Sharding setup time: {stats.sharding_time_seconds:.2f}s")
    logging.info(f"Estimated communication overhead: {stats.communication_overhead*100:.1f}%")
    
    # Log top sharded parameters by size
    if decisions:
        sorted_decisions = sorted(decisions, key=lambda x: x['size_mb'], reverse=True)
        logging.info("Top 5 sharded parameters by size:")
        for i, decision in enumerate(sorted_decisions[:5]):
            logging.info(f"  {i+1}. {decision['path']}: {decision['shape']} ({decision['size_mb']:.1f} MB)")
    
    logging.info("=" * 30)


def validate_sharding_performance(
    pytree,
    sharding_spec,
    mesh: jax.sharding.Mesh,
    config: ShardingConfig,
) -> Dict[str, Any]:
    """Validate sharding performance and detect potential issues.
    
    Args:
        pytree: The sharded pytree
        sharding_spec: The sharding specification used
        mesh: JAX mesh
        config: Sharding configuration
        
    Returns:
        Dictionary with validation results and performance metrics
    """
    validation_results = {
        "is_valid": True,
        "issues": [],
        "recommendations": [],
        "performance_metrics": {},
    }
    
    try:
        # Check for memory imbalance across devices
        memory_per_device = _estimate_memory_per_device(pytree, sharding_spec, mesh)
        
        if memory_per_device:
            max_memory = max(memory_per_device.values())
            min_memory = min(memory_per_device.values())
            memory_imbalance = (max_memory - min_memory) / max_memory if max_memory > 0 else 0
            
            validation_results["performance_metrics"]["memory_imbalance"] = memory_imbalance
            
            if memory_imbalance > 0.2:  # 20% imbalance threshold
                validation_results["issues"].append(
                    f"High memory imbalance across devices: {memory_imbalance*100:.1f}%"
                )
                validation_results["recommendations"].append(
                    "Consider adjusting sharding strategy to balance memory usage"
                )
        
        # Check for communication efficiency
        comm_efficiency = _estimate_communication_efficiency(sharding_spec, mesh)
        validation_results["performance_metrics"]["communication_efficiency"] = comm_efficiency
        
        if comm_efficiency < 0.7:  # 70% efficiency threshold
            validation_results["issues"].append(
                f"Low communication efficiency: {comm_efficiency*100:.1f}%"
            )
            validation_results["recommendations"].append(
                "Consider reducing the number of sharded parameters or adjusting mesh topology"
            )
        
        # Check for optimal sharding patterns
        sharding_patterns = _analyze_sharding_patterns(sharding_spec)
        validation_results["performance_metrics"]["sharding_patterns"] = sharding_patterns
        
        if sharding_patterns.get("suboptimal_count", 0) > 0:
            validation_results["recommendations"].append(
                f"Found {sharding_patterns['suboptimal_count']} suboptimal sharding patterns"
            )
        
        # Overall validation
        validation_results["is_valid"] = len(validation_results["issues"]) == 0
        
    except Exception as e:
        validation_results["is_valid"] = False
        validation_results["issues"].append(f"Validation failed: {e}")
    
    return validation_results


def _estimate_memory_per_device(
    pytree, 
    sharding_spec, 
    mesh: jax.sharding.Mesh
) -> Dict[int, float]:
    """Estimate memory usage per device."""
    # This is a simplified estimation
    # In practice, would need to analyze actual sharding distribution
    device_memory = {}
    total_devices = np.prod(mesh.shape)
    
    for i in range(total_devices):
        device_memory[i] = 0.0
    
    # For now, assume equal distribution (simplified)
    # Real implementation would analyze actual sharding specs
    return device_memory


def _estimate_communication_efficiency(sharding_spec, mesh: jax.sharding.Mesh) -> float:
    """Estimate communication efficiency based on sharding patterns."""
    # Simplified efficiency model
    # Real implementation would consider:
    # - Number of all-reduce operations needed
    # - Communication topology
    # - Parameter sizes and communication patterns
    
    fsdp_size = mesh.shape[FSDP_AXIS]
    
    if fsdp_size == 1:
        return 1.0  # No communication needed
    
    # Base efficiency decreases with more devices
    base_efficiency = 1.0 / np.sqrt(fsdp_size)
    
    return min(1.0, base_efficiency)


def _analyze_sharding_patterns(sharding_spec) -> Dict[str, Any]:
    """Analyze sharding patterns for optimization opportunities."""
    patterns = {
        "total_sharded": 0,
        "optimal_count": 0,
        "suboptimal_count": 0,
        "replicated_count": 0,
    }
    
    def analyze_spec(spec):
        if hasattr(spec, 'spec'):
            if any(axis == FSDP_AXIS for axis in spec.spec):
                patterns["total_sharded"] += 1
                # Simple heuristic: consider it optimal if sharding the first or last axis
                if spec.spec[0] == FSDP_AXIS or spec.spec[-1] == FSDP_AXIS:
                    patterns["optimal_count"] += 1
                else:
                    patterns["suboptimal_count"] += 1
            else:
                patterns["replicated_count"] += 1
    
    jax.tree_util.tree_map(analyze_spec, sharding_spec)
    
    return patterns
