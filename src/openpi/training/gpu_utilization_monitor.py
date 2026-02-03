"""GPU utilization monitoring system for OpenPI training pipeline."""

import dataclasses
import logging
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np


class BottleneckType(Enum):
    """Types of performance bottlenecks that can be detected."""
    DATA_LOADING = "data_loading"
    GPU_COMPUTATION = "gpu_computation"
    MEMORY_TRANSFER = "memory_transfer"
    GRADIENT_SYNC = "gradient_sync"
    COMPILATION = "compilation"
    NONE = "none"


@dataclasses.dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion based on detected bottlenecks."""
    bottleneck_type: BottleneckType
    message: str
    priority: int  # 1 = high, 2 = medium, 3 = low
    action_items: List[str]


@dataclasses.dataclass
class PerformanceMetrics:
    """Performance metrics collected during training."""
    timestamp: float
    gpu_utilization: float
    gpu_memory_usage: float
    gpu_memory_total: float
    data_loading_time: Optional[float] = None
    forward_pass_time: Optional[float] = None
    backward_pass_time: Optional[float] = None
    gradient_sync_time: Optional[float] = None
    batch_processing_rate: Optional[float] = None
    bottleneck_type: Optional[BottleneckType] = None


class GPUUtilizationMonitor:
    """Monitors GPU utilization and detects performance bottlenecks."""
    
    def __init__(
        self,
        target_utilization: float = 0.8,
        monitoring_interval: float = 1.0,
        metrics_window_size: int = 60,
        low_utilization_threshold: float = 0.5,
        high_utilization_threshold: float = 0.95,
    ):
        """Initialize GPU utilization monitor.
        
        Args:
            target_utilization: Target GPU utilization (0.0-1.0)
            monitoring_interval: How often to collect metrics (seconds)
            metrics_window_size: Number of recent metrics to keep for analysis
            low_utilization_threshold: Threshold below which utilization is considered low
            high_utilization_threshold: Threshold above which utilization is considered high
        """
        self.target_utilization = target_utilization
        self.monitoring_interval = monitoring_interval
        self.metrics_window_size = metrics_window_size
        self.low_utilization_threshold = low_utilization_threshold
        self.high_utilization_threshold = high_utilization_threshold
        
        # Metrics storage
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=metrics_window_size)
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self._step_start_time: Optional[float] = None
        self._data_loading_start_time: Optional[float] = None
        self._computation_start_time: Optional[float] = None
        
        # JAX device info
        self.devices = jax.devices()
        self.device_count = len(self.devices)
        
        logging.info(f"Initialized GPU monitor for {self.device_count} devices")
        logging.info(f"Target utilization: {target_utilization:.1%}")
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring:
            logging.warning("GPU monitoring already started")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logging.info("Started GPU utilization monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logging.info("Stopped GPU utilization monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self._monitoring:
            try:
                metrics = self._collect_gpu_metrics()
                # Use timeout to avoid blocking the main thread
                if self._lock.acquire(timeout=0.5):
                    try:
                        self.current_metrics = metrics
                        self.metrics_history.append(metrics)
                    finally:
                        self._lock.release()
                else:
                    # Skip this update if we can't get the lock quickly
                    logging.debug("GPU monitoring lock timeout, skipping update")
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logging.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_gpu_metrics(self) -> PerformanceMetrics:
        """Collect current GPU metrics."""
        # Get GPU memory info from JAX
        try:
            # Use JAX's memory info for the first device
            memory_info = jax.devices()[0].memory_stats()
            gpu_memory_usage = memory_info.get('bytes_in_use', 0)
            gpu_memory_total = memory_info.get('bytes_limit', 1)
            
            # Calculate memory usage as fraction
            memory_usage_fraction = gpu_memory_usage / max(gpu_memory_total, 1)
            
            # For GPU utilization, we'll use a heuristic based on memory usage
            # and computation activity. In a real implementation, you might use
            # nvidia-ml-py or similar for actual GPU utilization.
            gpu_utilization = min(memory_usage_fraction * 1.2, 1.0)
            
        except Exception as e:
            logging.warning(f"Failed to get GPU memory stats: {e}")
            gpu_memory_usage = 0
            gpu_memory_total = 1
            gpu_utilization = 0.0
        
        return PerformanceMetrics(
            timestamp=time.time(),
            gpu_utilization=gpu_utilization,
            gpu_memory_usage=gpu_memory_usage,
            gpu_memory_total=gpu_memory_total,
        )
    
    def get_current_utilization(self) -> float:
        """Get current GPU utilization."""
        with self._lock:
            if self.current_metrics is None:
                return 0.0
            return self.current_metrics.gpu_utilization
    
    def get_average_utilization(self, window_seconds: Optional[float] = None) -> float:
        """Get average GPU utilization over a time window."""
        # Copy metrics to avoid holding lock too long
        with self._lock:
            if not self.metrics_history:
                return 0.0
            metrics_copy = list(self.metrics_history)
        
        if window_seconds is None:
            # Use all available metrics
            metrics = metrics_copy
        else:
            # Filter metrics within the time window
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            metrics = [m for m in metrics_copy if m.timestamp >= cutoff_time]
        
        if not metrics:
            return 0.0
        
        return sum(m.gpu_utilization for m in metrics) / len(metrics)
    
    def get_memory_usage_stats(self) -> Dict[str, float]:
        """Get GPU memory usage statistics."""
        try:
            # Use timeout to avoid deadlock
            if self._lock.acquire(timeout=1.0):
                try:
                    if self.current_metrics is None:
                        return {"usage_bytes": 0, "total_bytes": 0, "usage_fraction": 0.0}
                    
                    return {
                        "usage_bytes": self.current_metrics.gpu_memory_usage,
                        "total_bytes": self.current_metrics.gpu_memory_total,
                        "usage_fraction": self.current_metrics.gpu_memory_usage / max(self.current_metrics.gpu_memory_total, 1),
                    }
                finally:
                    self._lock.release()
            else:
                # Timeout occurred, return default values
                logging.warning("GPU memory stats lock timeout, returning defaults")
                return {"usage_bytes": 0, "total_bytes": 0, "usage_fraction": 0.0}
        except Exception as e:
            logging.error(f"Error getting GPU memory stats: {e}")
            return {"usage_bytes": 0, "total_bytes": 0, "usage_fraction": 0.0}
    
    def detect_bottlenecks(self) -> List[BottleneckType]:
        """Detect performance bottlenecks based on recent metrics."""
        with self._lock:
            if len(self.metrics_history) < 10:  # Need sufficient data
                return []
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            avg_utilization = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            bottlenecks = []
            
            # Low GPU utilization suggests data loading or other bottlenecks
            if avg_utilization < self.low_utilization_threshold:
                bottlenecks.append(BottleneckType.DATA_LOADING)
            
            # Check for memory pressure
            memory_stats = self.get_memory_usage_stats()
            if memory_stats["usage_fraction"] > 0.9:
                bottlenecks.append(BottleneckType.MEMORY_TRANSFER)
            
            # Check for utilization variability (suggests compilation or sync issues)
            utilizations = [m.gpu_utilization for m in recent_metrics]
            if len(utilizations) > 1:
                std_dev = np.std(utilizations)
                if std_dev > 0.3:  # High variability
                    bottlenecks.append(BottleneckType.COMPILATION)
            
            return bottlenecks if bottlenecks else [BottleneckType.NONE]
    
    def suggest_optimizations(self) -> List[OptimizationSuggestion]:
        """Provide optimization suggestions based on detected bottlenecks."""
        bottlenecks = self.detect_bottlenecks()
        suggestions = []
        
        for bottleneck in bottlenecks:
            if bottleneck == BottleneckType.DATA_LOADING:
                suggestions.append(OptimizationSuggestion(
                    bottleneck_type=bottleneck,
                    message="Low GPU utilization detected, likely due to data loading bottlenecks",
                    priority=1,
                    action_items=[
                        "Increase num_workers in DataLoader configuration",
                        "Enable prefetch_factor > 2 in DataLoader",
                        "Use persistent_workers=True",
                        "Consider async video decoding for GR00T datasets",
                        "Check if data preprocessing can be optimized"
                    ]
                ))
            
            elif bottleneck == BottleneckType.MEMORY_TRANSFER:
                suggestions.append(OptimizationSuggestion(
                    bottleneck_type=bottleneck,
                    message="High memory usage detected, may cause performance issues",
                    priority=2,
                    action_items=[
                        "Reduce batch size to free up GPU memory",
                        "Enable gradient accumulation to maintain effective batch size",
                        "Use mixed precision training (bfloat16)",
                        "Consider model sharding with FSDP",
                        "Optimize data transfer patterns"
                    ]
                ))
            
            elif bottleneck == BottleneckType.COMPILATION:
                suggestions.append(OptimizationSuggestion(
                    bottleneck_type=bottleneck,
                    message="High utilization variability suggests compilation or synchronization issues",
                    priority=2,
                    action_items=[
                        "Warm up JIT compilation cache with sample data",
                        "Use static shapes to avoid recompilation",
                        "Check for gradient synchronization bottlenecks",
                        "Consider using compilation cache persistence",
                        "Profile JAX compilation times"
                    ]
                ))
        
        return suggestions
    
    def mark_step_start(self) -> None:
        """Mark the start of a training step for timing."""
        self._step_start_time = time.time()
    
    def mark_data_loading_start(self) -> None:
        """Mark the start of data loading for timing."""
        self._data_loading_start_time = time.time()
    
    def mark_data_loading_end(self) -> None:
        """Mark the end of data loading and record timing."""
        if self._data_loading_start_time is not None:
            data_loading_time = time.time() - self._data_loading_start_time
            with self._lock:
                if self.current_metrics is not None:
                    self.current_metrics = dataclasses.replace(
                        self.current_metrics,
                        data_loading_time=data_loading_time
                    )
    
    def mark_computation_start(self) -> None:
        """Mark the start of GPU computation for timing."""
        self._computation_start_time = time.time()
    
    def mark_computation_end(self) -> None:
        """Mark the end of GPU computation and record timing."""
        if self._computation_start_time is not None:
            computation_time = time.time() - self._computation_start_time
            with self._lock:
                if self.current_metrics is not None:
                    self.current_metrics = dataclasses.replace(
                        self.current_metrics,
                        forward_pass_time=computation_time
                    )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        # Copy data to avoid holding lock too long
        with self._lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            metrics_copy = list(self.metrics_history)
            current_metrics_copy = self.current_metrics
        
        recent_utilization = self.get_average_utilization(window_seconds=60)
        memory_stats = self.get_memory_usage_stats()
        bottlenecks = self.detect_bottlenecks()
        suggestions = self.suggest_optimizations()
        
        # Calculate utilization statistics
        utilizations = [m.gpu_utilization for m in metrics_copy]
        low_util_percentage = sum(1 for u in utilizations if u < 0.5) / len(utilizations) * 100
        
        return {
            "status": "active",
            "current_utilization": self.get_current_utilization(),
            "average_utilization_1min": recent_utilization,
            "target_utilization": self.target_utilization,
            "low_utilization_percentage": low_util_percentage,
            "memory_usage_fraction": memory_stats["usage_fraction"],
            "memory_usage_gb": memory_stats["usage_bytes"] / (1024**3),
            "memory_total_gb": memory_stats["total_bytes"] / (1024**3),
            "detected_bottlenecks": [b.value for b in bottlenecks],
            "optimization_suggestions": len(suggestions),
            "device_count": self.device_count,
            "metrics_collected": len(metrics_copy),
        }
    
    def log_performance_report(self, step: int) -> None:
        """Log a performance report to the logger."""
        summary = self.get_performance_summary()
        
        if summary["status"] == "no_data":
            logging.info(f"Step {step}: GPU monitoring - no data available yet")
            return
        
        # Format the report
        util_current = summary["current_utilization"]
        util_avg = summary["average_utilization_1min"]
        target = summary["target_utilization"]
        low_util_pct = summary["low_utilization_percentage"]
        mem_usage = summary["memory_usage_fraction"]
        
        status_msg = f"GPU: {util_current:.1%} current, {util_avg:.1%} avg (target: {target:.1%})"
        memory_msg = f"Memory: {mem_usage:.1%} used"
        
        if low_util_pct > 5:  # More than 5% of time with low utilization
            status_msg += f" ⚠️  Low utilization {low_util_pct:.1f}% of time"
        
        if summary["detected_bottlenecks"] and summary["detected_bottlenecks"] != ["none"]:
            bottlenecks_str = ", ".join(summary["detected_bottlenecks"])
            status_msg += f" | Bottlenecks: {bottlenecks_str}"
        
        logging.info(f"Step {step}: {status_msg} | {memory_msg}")
        
        # Log suggestions if any high-priority ones exist
        suggestions = self.suggest_optimizations()
        high_priority_suggestions = [s for s in suggestions if s.priority == 1]
        if high_priority_suggestions:
            for suggestion in high_priority_suggestions:
                logging.warning(f"Performance suggestion: {suggestion.message}")