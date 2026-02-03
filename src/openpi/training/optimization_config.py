"""
Optimization configuration system for GPU utilization and training performance.

This module provides a centralized configuration system for managing all GPU
utilization optimizations with validation and dynamic adjustment capabilities.
"""

import dataclasses
import logging
import os
import platform
import psutil
from typing import Any, Dict, List, Optional, Tuple, Union

import jax


@dataclasses.dataclass
class GPUMonitoringConfig:
    """Configuration for GPU utilization monitoring."""
    
    # Enable GPU utilization monitoring
    enable_gpu_monitoring: bool = True
    
    # Target GPU utilization threshold (0.0 to 1.0)
    target_gpu_utilization: float = 0.85
    
    # Monitoring interval in seconds
    monitoring_interval: float = 1.0
    
    # Enable automatic bottleneck detection
    enable_bottleneck_detection: bool = True
    
    # Enable automatic performance suggestions
    enable_performance_suggestions: bool = True
    
    # Enable wandb logging of performance metrics
    enable_wandb_logging: bool = True
    
    # Performance logging interval (in training steps)
    performance_log_interval: int = 100
    
    # Wandb logging interval (in training steps)
    wandb_log_interval: int = 10


@dataclasses.dataclass
class JAXOptimizationConfig:
    """Configuration for JAX compilation and memory optimizations."""
    
    # JIT compilation cache warming
    enable_jit_cache_warming: bool = True
    jit_warmup_iterations: int = 3
    jit_cache_persistence: bool = True
    jit_max_cache_size: int = 100
    log_compilation_timing: bool = True
    
    # Memory optimization settings
    enable_memory_optimization: bool = True
    target_memory_utilization: float = 0.9
    enable_auto_batch_sizing: bool = False  # Disabled by default for safety
    min_batch_size: int = 1
    max_batch_size: int = 512
    
    # Training step optimizations
    enable_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1
    enable_overlapped_computation: bool = True
    enable_mixed_precision: bool = True
    optimize_gradient_sync: bool = True


@dataclasses.dataclass
class DataLoadingConfig:
    """Configuration for data loading optimizations."""
    
    # Number of data loader workers (auto-detected if None)
    num_workers: Optional[int] = None
    
    # Prefetch factor for data loading
    prefetch_factor: int = 4
    
    # Enable persistent workers
    persistent_workers: bool = True
    
    # Enable memory pinning
    pin_memory: bool = True
    
    # Enable async video decoding for GR00T datasets
    enable_async_video_decoding: bool = True
    
    # Video frame cache size in MB
    video_frame_cache_size_mb: int = 256
    
    # Episode cache size per worker
    episode_cache_size: int = 16
    
    # Enable intelligent episode caching
    enable_intelligent_caching: bool = True
    
    # Enable parallel action processing
    enable_parallel_action_processing: bool = True
    
    # Number of action processing workers (auto-detected if None)
    action_processing_workers: Optional[int] = None


@dataclasses.dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training optimizations."""
    
    # Number of devices for FSDP sharding
    fsdp_devices: int = 1
    
    # Enable dynamic device count configuration
    enable_dynamic_device_count: bool = True
    
    # Log sharding decisions
    log_sharding_decisions: bool = True
    
    # Enable load balancing across devices
    enable_load_balancing: bool = True
    
    # Communication backend for gradient synchronization
    communication_backend: str = "nccl"
    
    # Minimum parameter size (in MB) to consider for sharding
    min_shard_size_mb: int = 4
    
    # Memory threshold for automatic sharding adjustments
    memory_threshold: float = 0.9


@dataclasses.dataclass
class OptimizationConfig:
    """
    Comprehensive optimization configuration for GPU utilization and training performance.
    
    This class manages all optimization settings with validation and dynamic adjustment
    based on system resources and hardware configuration.
    """
    
    # Sub-configurations for different optimization areas
    gpu_monitoring: GPUMonitoringConfig = dataclasses.field(default_factory=GPUMonitoringConfig)
    jax_optimization: JAXOptimizationConfig = dataclasses.field(default_factory=JAXOptimizationConfig)
    data_loading: DataLoadingConfig = dataclasses.field(default_factory=DataLoadingConfig)
    multi_gpu: MultiGPUConfig = dataclasses.field(default_factory=MultiGPUConfig)
    
    # Global optimization settings
    enable_all_optimizations: bool = True
    optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"
    
    # Hardware-specific overrides
    auto_detect_hardware: bool = True
    force_cpu_fallback: bool = False
    
    def __post_init__(self):
        """Validate and adjust configuration based on system resources."""
        if self.auto_detect_hardware:
            self._auto_detect_and_adjust()
        
        self._validate_configuration()
    
    def _auto_detect_and_adjust(self):
        """Automatically detect hardware and adjust configuration accordingly."""
        try:
            # Detect CPU cores and adjust worker counts
            cpu_count = os.cpu_count() or 4
            
            if self.data_loading.num_workers is None:
                # Use 75% of CPU cores for data loading, but cap at 8
                self.data_loading.num_workers = min(max(2, int(cpu_count * 0.75)), 8)
            
            if self.data_loading.action_processing_workers is None:
                # Use 50% of CPU cores for action processing, but cap at 4
                self.data_loading.action_processing_workers = min(max(1, int(cpu_count * 0.5)), 4)
            
            # Detect available memory and adjust cache sizes
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb < 16:
                # Low memory system - reduce cache sizes
                self.data_loading.video_frame_cache_size_mb = 128
                self.data_loading.episode_cache_size = 8
                self.jax_optimization.jit_max_cache_size = 50
                logging.info("Detected low memory system, reducing cache sizes")
            elif memory_gb > 64:
                # High memory system - increase cache sizes
                self.data_loading.video_frame_cache_size_mb = 512
                self.data_loading.episode_cache_size = 32
                self.jax_optimization.jit_max_cache_size = 200
                logging.info("Detected high memory system, increasing cache sizes")
            
            # Detect GPU configuration
            try:
                device_count = jax.device_count()
                if device_count > 1:
                    self.multi_gpu.fsdp_devices = min(device_count, self.multi_gpu.fsdp_devices)
                    logging.info(f"Detected {device_count} GPUs, adjusting FSDP configuration")
                else:
                    # Single GPU - disable some multi-GPU optimizations
                    self.multi_gpu.enable_load_balancing = False
                    self.jax_optimization.optimize_gradient_sync = False
            except Exception as e:
                logging.warning(f"Failed to detect GPU configuration: {e}")
                self.force_cpu_fallback = True
            
            # Adjust optimization level based on hardware
            if self.optimization_level == "balanced":
                if memory_gb < 16 or cpu_count < 8:
                    self._apply_conservative_settings()
                elif memory_gb > 64 and cpu_count > 16:
                    self._apply_aggressive_settings()
        
        except Exception as e:
            logging.warning(f"Hardware auto-detection failed: {e}")
            self._apply_conservative_settings()
    
    def _apply_conservative_settings(self):
        """Apply conservative optimization settings for lower-end hardware."""
        self.jax_optimization.enable_jit_cache_warming = True
        self.jax_optimization.jit_warmup_iterations = 2
        self.jax_optimization.enable_memory_optimization = True
        self.jax_optimization.enable_auto_batch_sizing = False
        self.jax_optimization.enable_gradient_accumulation = False
        self.jax_optimization.enable_overlapped_computation = False
        
        self.data_loading.prefetch_factor = 2
        self.data_loading.enable_async_video_decoding = True
        self.data_loading.enable_parallel_action_processing = False
        
        self.gpu_monitoring.monitoring_interval = 2.0
        
        logging.info("Applied conservative optimization settings")
    
    def _apply_aggressive_settings(self):
        """Apply aggressive optimization settings for high-end hardware."""
        self.jax_optimization.enable_jit_cache_warming = True
        self.jax_optimization.jit_warmup_iterations = 5
        self.jax_optimization.enable_memory_optimization = True
        self.jax_optimization.enable_auto_batch_sizing = True
        self.jax_optimization.enable_gradient_accumulation = True
        self.jax_optimization.gradient_accumulation_steps = 2
        self.jax_optimization.enable_overlapped_computation = True
        
        self.data_loading.prefetch_factor = 8
        self.data_loading.enable_async_video_decoding = True
        self.data_loading.enable_parallel_action_processing = True
        
        self.gpu_monitoring.monitoring_interval = 0.5
        
        logging.info("Applied aggressive optimization settings")
    
    def _validate_configuration(self):
        """Validate configuration settings and fix invalid values."""
        # Validate GPU monitoring settings
        self.gpu_monitoring.target_gpu_utilization = max(0.1, min(1.0, self.gpu_monitoring.target_gpu_utilization))
        self.gpu_monitoring.monitoring_interval = max(0.1, self.gpu_monitoring.monitoring_interval)
        
        # Validate JAX optimization settings
        self.jax_optimization.jit_warmup_iterations = max(1, self.jax_optimization.jit_warmup_iterations)
        self.jax_optimization.jit_max_cache_size = max(10, self.jax_optimization.jit_max_cache_size)
        self.jax_optimization.target_memory_utilization = max(0.5, min(0.95, self.jax_optimization.target_memory_utilization))
        self.jax_optimization.min_batch_size = max(1, self.jax_optimization.min_batch_size)
        self.jax_optimization.max_batch_size = max(self.jax_optimization.min_batch_size, self.jax_optimization.max_batch_size)
        self.jax_optimization.gradient_accumulation_steps = max(1, self.jax_optimization.gradient_accumulation_steps)
        
        # Validate data loading settings
        if self.data_loading.num_workers is not None:
            self.data_loading.num_workers = max(0, self.data_loading.num_workers)
        self.data_loading.prefetch_factor = max(1, self.data_loading.prefetch_factor)
        self.data_loading.video_frame_cache_size_mb = max(64, self.data_loading.video_frame_cache_size_mb)
        self.data_loading.episode_cache_size = max(1, self.data_loading.episode_cache_size)
        if self.data_loading.action_processing_workers is not None:
            self.data_loading.action_processing_workers = max(1, self.data_loading.action_processing_workers)
        
        # Validate multi-GPU settings
        self.multi_gpu.fsdp_devices = max(1, self.multi_gpu.fsdp_devices)
        self.multi_gpu.min_shard_size_mb = max(1, self.multi_gpu.min_shard_size_mb)
        self.multi_gpu.memory_threshold = max(0.5, min(0.95, self.multi_gpu.memory_threshold))
        
        # Disable optimizations if force_cpu_fallback is True
        if self.force_cpu_fallback:
            self.gpu_monitoring.enable_gpu_monitoring = False
            self.jax_optimization.enable_memory_optimization = False
            self.jax_optimization.enable_auto_batch_sizing = False
            self.multi_gpu.enable_load_balancing = False
    
    def get_effective_config_dict(self) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary for logging."""
        return {
            "optimization_level": self.optimization_level,
            "enable_all_optimizations": self.enable_all_optimizations,
            "auto_detect_hardware": self.auto_detect_hardware,
            "force_cpu_fallback": self.force_cpu_fallback,
            "gpu_monitoring": dataclasses.asdict(self.gpu_monitoring),
            "jax_optimization": dataclasses.asdict(self.jax_optimization),
            "data_loading": dataclasses.asdict(self.data_loading),
            "multi_gpu": dataclasses.asdict(self.multi_gpu),
        }
    
    def disable_all_optimizations(self):
        """Disable all optimizations for debugging or fallback scenarios."""
        self.enable_all_optimizations = False
        
        self.gpu_monitoring.enable_gpu_monitoring = False
        self.gpu_monitoring.enable_bottleneck_detection = False
        self.gpu_monitoring.enable_performance_suggestions = False
        
        self.jax_optimization.enable_jit_cache_warming = False
        self.jax_optimization.enable_memory_optimization = False
        self.jax_optimization.enable_auto_batch_sizing = False
        self.jax_optimization.enable_gradient_accumulation = False
        self.jax_optimization.enable_overlapped_computation = False
        self.jax_optimization.enable_mixed_precision = False
        self.jax_optimization.optimize_gradient_sync = False
        
        self.data_loading.enable_async_video_decoding = False
        self.data_loading.enable_intelligent_caching = False
        self.data_loading.enable_parallel_action_processing = False
        
        self.multi_gpu.enable_dynamic_device_count = False
        self.multi_gpu.enable_load_balancing = False
        
        logging.info("All optimizations disabled")
    
    @classmethod
    def create_for_hardware_setup(cls, setup_type: str = "auto") -> "OptimizationConfig":
        """
        Create an optimization configuration for a specific hardware setup.
        
        Args:
            setup_type: "auto", "single_gpu", "multi_gpu", "cpu_only", "debug"
        """
        config = cls()
        
        if setup_type == "single_gpu":
            config.multi_gpu.fsdp_devices = 1
            config.multi_gpu.enable_load_balancing = False
            config.jax_optimization.optimize_gradient_sync = False
            config.optimization_level = "balanced"
        
        elif setup_type == "multi_gpu":
            config.multi_gpu.enable_load_balancing = True
            config.jax_optimization.optimize_gradient_sync = True
            config.jax_optimization.enable_gradient_accumulation = True
            config.optimization_level = "aggressive"
        
        elif setup_type == "cpu_only":
            config.force_cpu_fallback = True
            config.gpu_monitoring.enable_gpu_monitoring = False
            config.jax_optimization.enable_memory_optimization = False
            config.optimization_level = "conservative"
        
        elif setup_type == "debug":
            config.disable_all_optimizations()
            config.gpu_monitoring.enable_wandb_logging = False
        
        elif setup_type == "auto":
            # Use default auto-detection
            pass
        
        else:
            raise ValueError(f"Unknown setup_type: {setup_type}")
        
        return config


def create_optimization_config_from_train_config(train_config) -> OptimizationConfig:
    """
    Create an OptimizationConfig from a TrainConfig with backward compatibility.
    
    This function extracts optimization settings from the existing TrainConfig
    and creates a comprehensive OptimizationConfig.
    """
    config = OptimizationConfig()
    
    # Extract GPU monitoring settings
    config.gpu_monitoring.enable_gpu_monitoring = getattr(train_config, 'enable_gpu_monitoring', True)
    config.gpu_monitoring.target_gpu_utilization = getattr(train_config, 'target_gpu_utilization', 0.85)
    config.gpu_monitoring.monitoring_interval = getattr(train_config, 'gpu_monitoring_interval', 1.0)
    config.gpu_monitoring.enable_bottleneck_detection = getattr(train_config, 'enable_bottleneck_detection', True)
    config.gpu_monitoring.enable_performance_suggestions = getattr(train_config, 'enable_performance_suggestions', True)
    config.gpu_monitoring.enable_wandb_logging = getattr(train_config, 'wandb_enabled', True)
    config.gpu_monitoring.performance_log_interval = getattr(train_config, 'log_interval', 100)
    config.gpu_monitoring.wandb_log_interval = max(10, getattr(train_config, 'log_interval', 100) // 10)
    
    # Extract JAX optimization settings
    config.jax_optimization.enable_jit_cache_warming = getattr(train_config, 'enable_jit_cache_warming', True)
    config.jax_optimization.jit_warmup_iterations = getattr(train_config, 'jit_warmup_iterations', 3)
    config.jax_optimization.jit_cache_persistence = getattr(train_config, 'jit_cache_persistence', True)
    config.jax_optimization.jit_max_cache_size = getattr(train_config, 'jit_max_cache_size', 100)
    config.jax_optimization.log_compilation_timing = getattr(train_config, 'log_compilation_timing', True)
    config.jax_optimization.enable_memory_optimization = getattr(train_config, 'enable_memory_optimization', True)
    config.jax_optimization.target_memory_utilization = getattr(train_config, 'target_memory_utilization', 0.9)
    config.jax_optimization.enable_auto_batch_sizing = getattr(train_config, 'enable_auto_batch_sizing', False)
    config.jax_optimization.min_batch_size = getattr(train_config, 'min_batch_size', 1)
    config.jax_optimization.max_batch_size = getattr(train_config, 'max_batch_size', 512)
    config.jax_optimization.enable_gradient_accumulation = getattr(train_config, 'enable_gradient_accumulation', False)
    config.jax_optimization.gradient_accumulation_steps = getattr(train_config, 'gradient_accumulation_steps', 1)
    config.jax_optimization.enable_overlapped_computation = getattr(train_config, 'enable_overlapped_computation', True)
    config.jax_optimization.enable_mixed_precision = getattr(train_config, 'enable_mixed_precision', True)
    config.jax_optimization.optimize_gradient_sync = getattr(train_config, 'optimize_gradient_sync', True)
    
    # Extract data loading settings
    config.data_loading.num_workers = getattr(train_config, 'num_workers', None)
    
    # Extract multi-GPU settings
    config.multi_gpu.fsdp_devices = getattr(train_config, 'fsdp_devices', 1)
    
    return config