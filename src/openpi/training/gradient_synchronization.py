"""Efficient gradient synchronization for JAX multi-GPU training."""

import dataclasses
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


@dataclasses.dataclass
class GradientSyncConfig:
    """Configuration for gradient synchronization."""
    
    # Synchronization strategy: "allreduce", "async", "hierarchical"
    sync_strategy: str = "allreduce"
    
    # Whether to enable gradient compression
    enable_compression: bool = False
    
    # Compression method: "quantization", "sparsification", "none"
    compression_method: str = "quantization"
    
    # Compression ratio (for quantization/sparsification)
    compression_ratio: float = 0.1
    
    # Whether to enable asynchronous communication
    enable_async_comm: bool = True
    
    # Communication backend: "nccl", "gloo"
    comm_backend: str = "nccl"
    
    # Whether to overlap communication with computation
    enable_overlap: bool = True
    
    # Gradient accumulation steps before synchronization
    sync_frequency: int = 1
    
    # Whether to enable gradient clipping before sync
    enable_gradient_clipping: bool = True
    
    # Gradient clipping threshold
    gradient_clip_threshold: float = 1.0
    
    # Whether to log synchronization statistics
    log_sync_stats: bool = True


@dataclasses.dataclass
class SyncStats:
    """Statistics for gradient synchronization."""
    
    sync_time_seconds: float = 0.0
    compression_time_seconds: float = 0.0
    communication_time_seconds: float = 0.0
    decompression_time_seconds: float = 0.0
    total_gradients: int = 0
    compressed_gradients: int = 0
    compression_ratio_achieved: float = 0.0
    communication_volume_mb: float = 0.0
    sync_frequency_actual: int = 1
    gradient_norm_before: float = 0.0
    gradient_norm_after: float = 0.0


class GradientCompressor:
    """Handles gradient compression for efficient communication."""
    
    def __init__(self, config: GradientSyncConfig):
        self.config = config
        self._compression_stats = {}
        
    def compress_gradients(
        self, 
        gradients: at.Params
    ) -> Tuple[at.Params, Dict[str, Any]]:
        """Compress gradients for efficient communication.
        
        Args:
            gradients: Gradients to compress
            
        Returns:
            Tuple of (compressed_gradients, compression_metadata)
        """
        if not self.config.enable_compression:
            return gradients, {}
        
        start_time = time.time()
        
        if self.config.compression_method == "quantization":
            compressed_grads, metadata = self._quantize_gradients(gradients)
        elif self.config.compression_method == "sparsification":
            compressed_grads, metadata = self._sparsify_gradients(gradients)
        else:
            compressed_grads, metadata = gradients, {}
        
        compression_time = time.time() - start_time
        metadata["compression_time"] = compression_time
        
        return compressed_grads, metadata
    
    def decompress_gradients(
        self, 
        compressed_gradients: at.Params, 
        metadata: Dict[str, Any]
    ) -> at.Params:
        """Decompress gradients after communication.
        
        Args:
            compressed_gradients: Compressed gradients
            metadata: Compression metadata
            
        Returns:
            Decompressed gradients
        """
        if not self.config.enable_compression or not metadata:
            return compressed_gradients
        
        if self.config.compression_method == "quantization":
            return self._dequantize_gradients(compressed_gradients, metadata)
        elif self.config.compression_method == "sparsification":
            return self._desparsify_gradients(compressed_gradients, metadata)
        else:
            return compressed_gradients
    
    def _quantize_gradients(
        self, 
        gradients: at.Params
    ) -> Tuple[at.Params, Dict[str, Any]]:
        """Quantize gradients to reduce communication volume."""
        quantized_grads = {}
        scales = {}
        
        def quantize_array(grad_array):
            if grad_array.size == 0:
                return grad_array, 1.0
            
            # Compute scale for quantization
            abs_max = jnp.max(jnp.abs(grad_array))
            scale = abs_max / (2**(8 - 1) - 1)  # 8-bit quantization
            
            if scale == 0:
                return jnp.zeros_like(grad_array, dtype=jnp.int8), 1.0
            
            # Quantize to int8
            quantized = jnp.round(grad_array / scale).astype(jnp.int8)
            
            return quantized, scale
        
        # Apply quantization to each gradient array
        for key, grad in gradients.items():
            if isinstance(grad, jax.Array) and grad.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
                quantized_grads[key], scales[key] = quantize_array(grad)
            else:
                quantized_grads[key] = grad
                scales[key] = 1.0
        
        metadata = {
            "method": "quantization",
            "scales": scales,
            "original_dtypes": {k: v.dtype for k, v in gradients.items() if isinstance(v, jax.Array)},
        }
        
        return quantized_grads, metadata
    
    def _dequantize_gradients(
        self, 
        quantized_gradients: at.Params, 
        metadata: Dict[str, Any]
    ) -> at.Params:
        """Dequantize gradients after communication."""
        scales = metadata.get("scales", {})
        original_dtypes = metadata.get("original_dtypes", {})
        
        dequantized_grads = {}
        
        for key, quantized_grad in quantized_gradients.items():
            if key in scales and isinstance(quantized_grad, jax.Array):
                scale = scales[key]
                original_dtype = original_dtypes.get(key, jnp.float32)
                
                # Dequantize back to original dtype
                dequantized = (quantized_grad.astype(original_dtype) * scale)
                dequantized_grads[key] = dequantized
            else:
                dequantized_grads[key] = quantized_grad
        
        return dequantized_grads
    
    def _sparsify_gradients(
        self, 
        gradients: at.Params
    ) -> Tuple[at.Params, Dict[str, Any]]:
        """Sparsify gradients by keeping only top-k elements."""
        sparsified_grads = {}
        sparsification_masks = {}
        
        def sparsify_array(grad_array):
            if grad_array.size == 0:
                return grad_array, None
            
            # Flatten for easier processing
            flat_grad = grad_array.flatten()
            
            # Keep top-k elements by magnitude
            k = max(1, int(flat_grad.size * self.config.compression_ratio))
            
            # Get indices of top-k elements
            abs_grad = jnp.abs(flat_grad)
            _, top_k_indices = jax.lax.top_k(abs_grad, k)
            
            # Create sparse representation
            sparse_values = flat_grad[top_k_indices]
            
            # Create mask for reconstruction
            mask = jnp.zeros_like(flat_grad, dtype=bool)
            mask = mask.at[top_k_indices].set(True)
            
            return sparse_values, mask.reshape(grad_array.shape)
        
        for key, grad in gradients.items():
            if isinstance(grad, jax.Array) and grad.dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
                sparse_values, mask = sparsify_array(grad)
                sparsified_grads[key] = sparse_values
                sparsification_masks[key] = mask
            else:
                sparsified_grads[key] = grad
                sparsification_masks[key] = None
        
        metadata = {
            "method": "sparsification",
            "masks": sparsification_masks,
            "original_shapes": {k: v.shape for k, v in gradients.items() if isinstance(v, jax.Array)},
        }
        
        return sparsified_grads, metadata
    
    def _desparsify_gradients(
        self, 
        sparsified_gradients: at.Params, 
        metadata: Dict[str, Any]
    ) -> at.Params:
        """Reconstruct gradients from sparse representation."""
        masks = metadata.get("masks", {})
        original_shapes = metadata.get("original_shapes", {})
        
        desparsified_grads = {}
        
        for key, sparse_grad in sparsified_gradients.items():
            if key in masks and masks[key] is not None:
                mask = masks[key]
                original_shape = original_shapes.get(key, sparse_grad.shape)
                
                # Reconstruct full gradient
                full_grad = jnp.zeros(original_shape, dtype=sparse_grad.dtype)
                full_grad = jnp.where(mask, sparse_grad, 0.0)
                
                desparsified_grads[key] = full_grad
            else:
                desparsified_grads[key] = sparse_grad
        
        return desparsified_grads


class AsyncGradientSynchronizer:
    """Handles asynchronous gradient synchronization with overlap."""
    
    def __init__(self, config: GradientSyncConfig, mesh: jax.sharding.Mesh):
        self.config = config
        self.mesh = mesh
        self.compressor = GradientCompressor(config)
        self._sync_stats = SyncStats()
        self._accumulated_gradients = None
        self._accumulation_count = 0
        
    def synchronize_gradients(
        self, 
        gradients: at.Params,
        step: int,
    ) -> Tuple[at.Params, SyncStats]:
        """Synchronize gradients across devices with optional compression and async communication.
        
        Args:
            gradients: Local gradients to synchronize
            step: Current training step
            
        Returns:
            Tuple of (synchronized_gradients, sync_statistics)
        """
        start_time = time.time()
        
        # Reset stats for this sync
        sync_stats = SyncStats()
        sync_stats.total_gradients = len(gradients)
        sync_stats.gradient_norm_before = optax.global_norm(gradients)
        
        # Apply gradient clipping if enabled
        if self.config.enable_gradient_clipping:
            gradients = self._clip_gradients(gradients)
        
        # Handle gradient accumulation
        if self.config.sync_frequency > 1:
            gradients, should_sync = self._accumulate_gradients(gradients)
            if not should_sync:
                # Return accumulated gradients without synchronization
                sync_stats.sync_frequency_actual = self._accumulation_count
                return gradients, sync_stats
        
        # Compress gradients if enabled
        compression_start = time.time()
        compressed_grads, compression_metadata = self.compressor.compress_gradients(gradients)
        sync_stats.compression_time_seconds = time.time() - compression_start
        
        if compression_metadata:
            sync_stats.compressed_gradients = len(compressed_grads)
            sync_stats.compression_ratio_achieved = self._calculate_compression_ratio(
                gradients, compressed_grads
            )
        
        # Synchronize gradients based on strategy
        comm_start = time.time()
        
        if self.config.sync_strategy == "allreduce":
            synchronized_grads = self._allreduce_sync(compressed_grads)
        elif self.config.sync_strategy == "async":
            synchronized_grads = self._async_sync(compressed_grads)
        elif self.config.sync_strategy == "hierarchical":
            synchronized_grads = self._hierarchical_sync(compressed_grads)
        else:
            raise ValueError(f"Unknown sync strategy: {self.config.sync_strategy}")
        
        sync_stats.communication_time_seconds = time.time() - comm_start
        
        # Decompress gradients if needed
        decompression_start = time.time()
        final_grads = self.compressor.decompress_gradients(synchronized_grads, compression_metadata)
        sync_stats.decompression_time_seconds = time.time() - decompression_start
        
        # Calculate final statistics
        sync_stats.sync_time_seconds = time.time() - start_time
        sync_stats.gradient_norm_after = optax.global_norm(final_grads)
        sync_stats.communication_volume_mb = self._estimate_communication_volume(compressed_grads)
        sync_stats.sync_frequency_actual = self.config.sync_frequency
        
        # Log statistics if enabled
        if self.config.log_sync_stats and step % 100 == 0:  # Log every 100 steps
            self._log_sync_statistics(sync_stats, step)
        
        return final_grads, sync_stats
    
    def _clip_gradients(self, gradients: at.Params) -> at.Params:
        """Apply gradient clipping."""
        grad_norm = optax.global_norm(gradients)
        
        if grad_norm > self.config.gradient_clip_threshold:
            # Scale gradients to threshold
            scale_factor = self.config.gradient_clip_threshold / grad_norm
            clipped_grads = jax.tree.map(lambda g: g * scale_factor, gradients)
            return clipped_grads
        
        return gradients
    
    def _accumulate_gradients(self, gradients: at.Params) -> Tuple[at.Params, bool]:
        """Accumulate gradients over multiple steps."""
        if self._accumulated_gradients is None:
            self._accumulated_gradients = gradients
        else:
            self._accumulated_gradients = jax.tree.map(
                lambda acc, new: acc + new, self._accumulated_gradients, gradients
            )
        
        self._accumulation_count += 1
        
        if self._accumulation_count >= self.config.sync_frequency:
            # Time to synchronize
            accumulated = self._accumulated_gradients
            
            # Average the accumulated gradients
            accumulated = jax.tree.map(
                lambda g: g / self.config.sync_frequency, accumulated
            )
            
            # Reset accumulation
            self._accumulated_gradients = None
            self._accumulation_count = 0
            
            return accumulated, True
        
        return self._accumulated_gradients, False
    
    def _allreduce_sync(self, gradients: at.Params) -> at.Params:
        """Synchronize gradients using all-reduce."""
        # Use JAX's built-in all-reduce with proper sharding
        def allreduce_fn(grad):
            if isinstance(grad, jax.Array):
                # Apply all-reduce across FSDP axis
                return jax.lax.pmean(grad, axis_name=sharding.FSDP_AXIS)
            return grad
        
        # Apply all-reduce with mesh context
        with sharding.set_mesh(self.mesh):
            synchronized_grads = jax.tree.map(allreduce_fn, gradients)
        
        return synchronized_grads
    
    def _async_sync(self, gradients: at.Params) -> at.Params:
        """Asynchronous gradient synchronization with communication overlap."""
        # For JAX, async communication is handled automatically by the runtime
        # We can use the same all-reduce but with async hints
        
        def async_allreduce_fn(grad):
            if isinstance(grad, jax.Array):
                # JAX automatically handles async communication when possible
                return jax.lax.pmean(grad, axis_name=sharding.FSDP_AXIS)
            return grad
        
        with sharding.set_mesh(self.mesh):
            synchronized_grads = jax.tree.map(async_allreduce_fn, gradients)
        
        return synchronized_grads
    
    def _hierarchical_sync(self, gradients: at.Params) -> at.Params:
        """Hierarchical gradient synchronization for large-scale training."""
        # Hierarchical sync: first reduce within nodes, then across nodes
        # For simplicity, fall back to all-reduce for now
        # In practice, this would implement a two-stage reduction
        
        return self._allreduce_sync(gradients)
    
    def _calculate_compression_ratio(
        self, 
        original_grads: at.Params, 
        compressed_grads: at.Params
    ) -> float:
        """Calculate achieved compression ratio."""
        try:
            original_size = sum(
                grad.size * grad.itemsize 
                for grad in jax.tree_util.tree_leaves(original_grads)
                if isinstance(grad, jax.Array)
            )
            
            compressed_size = sum(
                grad.size * grad.itemsize 
                for grad in jax.tree_util.tree_leaves(compressed_grads)
                if isinstance(grad, jax.Array)
            )
            
            if original_size > 0:
                return compressed_size / original_size
            
        except Exception as e:
            logging.debug(f"Failed to calculate compression ratio: {e}")
        
        return 1.0
    
    def _estimate_communication_volume(self, gradients: at.Params) -> float:
        """Estimate communication volume in MB."""
        try:
            total_bytes = sum(
                grad.size * grad.itemsize 
                for grad in jax.tree_util.tree_leaves(gradients)
                if isinstance(grad, jax.Array)
            )
            return total_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    def _log_sync_statistics(self, stats: SyncStats, step: int):
        """Log gradient synchronization statistics."""
        logging.info(f"Gradient Sync Stats (Step {step}):")
        logging.info(f"  Total sync time: {stats.sync_time_seconds:.3f}s")
        logging.info(f"  Compression time: {stats.compression_time_seconds:.3f}s")
        logging.info(f"  Communication time: {stats.communication_time_seconds:.3f}s")
        logging.info(f"  Decompression time: {stats.decompression_time_seconds:.3f}s")
        logging.info(f"  Gradient norm: {stats.gradient_norm_before:.4f} -> {stats.gradient_norm_after:.4f}")
        logging.info(f"  Communication volume: {stats.communication_volume_mb:.2f} MB")
        
        if stats.compressed_gradients > 0:
            logging.info(f"  Compression ratio: {stats.compression_ratio_achieved:.3f}")
        
        if stats.sync_frequency_actual > 1:
            logging.info(f"  Sync frequency: {stats.sync_frequency_actual}")
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get comprehensive synchronization statistics."""
        return {
            "config": {
                "sync_strategy": self.config.sync_strategy,
                "enable_compression": self.config.enable_compression,
                "compression_method": self.config.compression_method,
                "enable_async_comm": self.config.enable_async_comm,
                "sync_frequency": self.config.sync_frequency,
            },
            "mesh_info": {
                "shape": self.mesh.shape,
                "axis_names": self.mesh.axis_names,
                "device_count": len(self.mesh.devices.flat),
            },
            "current_stats": dataclasses.asdict(self._sync_stats),
        }


def create_gradient_sync_function(
    config: GradientSyncConfig,
    mesh: jax.sharding.Mesh,
) -> Callable[[at.Params, int], Tuple[at.Params, SyncStats]]:
    """Create a JIT-compiled gradient synchronization function.
    
    Args:
        config: Gradient synchronization configuration
        mesh: JAX mesh for multi-device coordination
        
    Returns:
        JIT-compiled gradient synchronization function
    """
    synchronizer = AsyncGradientSynchronizer(config, mesh)
    
    # Create JIT-compiled sync function
    @functools.partial(
        jax.jit,
        static_argnums=(1,),  # step is static
    )
    def sync_gradients_jit(gradients: at.Params, step: int) -> Tuple[at.Params, Dict[str, float]]:
        """JIT-compiled gradient synchronization."""
        
        # Apply gradient clipping if enabled
        if config.enable_gradient_clipping:
            grad_norm = optax.global_norm(gradients)
            clip_factor = jnp.minimum(1.0, config.gradient_clip_threshold / grad_norm)
            gradients = jax.tree.map(lambda g: g * clip_factor, gradients)
        
        # Synchronize using all-reduce (simplified for JIT compilation)
        def sync_fn(grad):
            if isinstance(grad, jax.Array):
                return jax.lax.pmean(grad, axis_name=sharding.FSDP_AXIS)
            return grad
        
        with sharding.set_mesh(mesh):
            synchronized_grads = jax.tree.map(sync_fn, gradients)
        
        # Return basic stats (limited by JIT constraints)
        stats_dict = {
            "gradient_norm": optax.global_norm(synchronized_grads),
            "sync_step": float(step),
        }
        
        return synchronized_grads, stats_dict
    
    return sync_gradients_jit


def optimize_gradient_communication(
    gradients: at.Params,
    mesh: jax.sharding.Mesh,
    config: GradientSyncConfig,
) -> at.Params:
    """Optimize gradient communication patterns for better performance.
    
    Args:
        gradients: Gradients to optimize
        mesh: JAX mesh
        config: Synchronization configuration
        
    Returns:
        Optimized gradients with efficient communication patterns
    """
    # Optimization strategies:
    # 1. Bucket small gradients together
    # 2. Prioritize large gradients for early communication
    # 3. Use appropriate data types for communication
    
    optimized_grads = {}
    
    # Group gradients by size for bucketing
    small_grads = {}
    large_grads = {}
    
    size_threshold = 1024 * 1024  # 1MB threshold
    
    for key, grad in gradients.items():
        if isinstance(grad, jax.Array):
            grad_size = grad.size * grad.itemsize
            
            if grad_size < size_threshold:
                small_grads[key] = grad
            else:
                large_grads[key] = grad
        else:
            optimized_grads[key] = grad
    
    # Process large gradients first (they benefit most from optimization)
    for key, grad in large_grads.items():
        # Ensure optimal data layout for communication
        if grad.dtype == jnp.float32 and config.enable_compression:
            # Convert to bfloat16 for communication efficiency
            optimized_grads[key] = grad.astype(jnp.bfloat16)
        else:
            optimized_grads[key] = grad
    
    # Bucket small gradients (simplified - in practice would concatenate)
    for key, grad in small_grads.items():
        optimized_grads[key] = grad
    
    return optimized_grads