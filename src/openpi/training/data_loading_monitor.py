"""
Data loading performance monitoring for OpenPI training.

This module provides comprehensive monitoring of data loading performance to identify
bottlenecks and optimize GPU utilization during training.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import statistics

import numpy as np


@dataclass
class DataLoadingMetrics:
    """Metrics for data loading performance."""
    batch_load_time: float = 0.0
    batch_size: int = 0
    worker_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    video_decode_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    queue_size: int = 0
    
    @property
    def samples_per_second(self) -> float:
        """Calculate samples processed per second."""
        if self.batch_load_time > 0:
            return self.batch_size / self.batch_load_time
        return 0.0


@dataclass
class BottleneckAnalysis:
    """Analysis of data loading bottlenecks."""
    bottleneck_type: str
    severity: str  # "low", "medium", "high"
    description: str
    recommendations: List[str]
    metrics: Dict[str, float]


class DataLoadingMonitor:
    """Monitor data loading performance and detect bottlenecks."""
    
    def __init__(
        self,
        window_size: int = 100,
        log_interval: int = 50,
        bottleneck_threshold: float = 0.1,  # seconds
    ):
        """Initialize data loading monitor.
        
        Args:
            window_size: Number of recent batches to keep for analysis
            log_interval: How often to log performance metrics (in batches)
            bottleneck_threshold: Threshold for detecting slow batch loading (seconds)
        """
        self.window_size = window_size
        self.log_interval = log_interval
        self.bottleneck_threshold = bottleneck_threshold
        
        # Metrics storage
        self._metrics_history = deque(maxlen=window_size)
        self._worker_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
        # Counters
        self._batch_count = 0
        self._total_batches = 0
        self._start_time = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Bottleneck detection
        self._bottlenecks = []
        self._last_analysis_time = time.time()
        
        # Logger
        self._logger = logging.getLogger(__name__)
    
    def record_batch_metrics(self, metrics: DataLoadingMetrics):
        """Record metrics for a single batch."""
        with self._lock:
            self._metrics_history.append(metrics)
            
            if metrics.worker_id is not None:
                self._worker_metrics[metrics.worker_id].append(metrics)
            
            self._batch_count += 1
            self._total_batches += 1
            
            # Log periodically
            if self._batch_count >= self.log_interval:
                self._log_performance_summary()
                self._batch_count = 0
            
            # Analyze bottlenecks periodically
            if time.time() - self._last_analysis_time > 30.0:  # Every 30 seconds
                self._analyze_bottlenecks()
                self._last_analysis_time = time.time()
    
    def _log_performance_summary(self):
        """Log performance summary."""
        if not self._metrics_history:
            return
        
        recent_metrics = list(self._metrics_history)[-self.log_interval:]
        
        # Calculate statistics
        load_times = [m.batch_load_time for m in recent_metrics]
        samples_per_sec = [m.samples_per_second for m in recent_metrics]
        video_decode_times = [m.video_decode_time for m in recent_metrics if m.video_decode_time > 0]
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate >= 0]
        
        avg_load_time = statistics.mean(load_times) if load_times else 0
        avg_samples_per_sec = statistics.mean(samples_per_sec) if samples_per_sec else 0
        avg_video_decode_time = statistics.mean(video_decode_times) if video_decode_times else 0
        avg_cache_hit_rate = statistics.mean(cache_hit_rates) if cache_hit_rates else 0
        
        # Log summary
        self._logger.info(
            f"Data Loading Performance (last {len(recent_metrics)} batches): "
            f"avg_load_time={avg_load_time:.4f}s, "
            f"avg_samples_per_sec={avg_samples_per_sec:.1f}, "
            f"avg_video_decode_time={avg_video_decode_time:.4f}s, "
            f"avg_cache_hit_rate={avg_cache_hit_rate:.2%}"
        )
        
        # Check for slow batches
        slow_batches = [m for m in recent_metrics if m.batch_load_time > self.bottleneck_threshold]
        if slow_batches:
            self._logger.warning(
                f"Detected {len(slow_batches)} slow batches (>{self.bottleneck_threshold}s) "
                f"in last {len(recent_metrics)} batches"
            )
    
    def _analyze_bottlenecks(self):
        """Analyze recent metrics to detect bottlenecks."""
        if len(self._metrics_history) < 10:
            return
        
        recent_metrics = list(self._metrics_history)
        bottlenecks = []
        
        # Analyze batch loading times
        load_times = [m.batch_load_time for m in recent_metrics]
        avg_load_time = statistics.mean(load_times)
        max_load_time = max(load_times)
        
        if avg_load_time > self.bottleneck_threshold:
            severity = "high" if avg_load_time > self.bottleneck_threshold * 3 else "medium"
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type="slow_data_loading",
                severity=severity,
                description=f"Average batch loading time ({avg_load_time:.4f}s) exceeds threshold",
                recommendations=[
                    "Increase number of data loader workers",
                    "Enable persistent workers",
                    "Increase prefetch factor",
                    "Check disk I/O performance",
                    "Enable async video decoding",
                ],
                metrics={
                    "avg_load_time": avg_load_time,
                    "max_load_time": max_load_time,
                    "threshold": self.bottleneck_threshold,
                }
            ))
        
        # Analyze video decoding performance
        video_decode_times = [m.video_decode_time for m in recent_metrics if m.video_decode_time > 0]
        if video_decode_times:
            avg_video_time = statistics.mean(video_decode_times)
            video_ratio = avg_video_time / avg_load_time if avg_load_time > 0 else 0
            
            if video_ratio > 0.5:  # Video decoding takes >50% of loading time
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type="slow_video_decoding",
                    severity="medium" if video_ratio > 0.7 else "low",
                    description=f"Video decoding takes {video_ratio:.1%} of batch loading time",
                    recommendations=[
                        "Enable async video decoding",
                        "Increase video cache size",
                        "Use faster video backend (torchcodec vs pyav)",
                        "Reduce video resolution if possible",
                    ],
                    metrics={
                        "avg_video_decode_time": avg_video_time,
                        "video_time_ratio": video_ratio,
                    }
                ))
        
        # Analyze cache performance
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate >= 0]
        if cache_hit_rates:
            avg_cache_hit_rate = statistics.mean(cache_hit_rates)
            
            if avg_cache_hit_rate < 0.3:  # Low cache hit rate
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type="low_cache_hit_rate",
                    severity="medium" if avg_cache_hit_rate < 0.1 else "low",
                    description=f"Low cache hit rate ({avg_cache_hit_rate:.1%})",
                    recommendations=[
                        "Increase episode cache size",
                        "Increase video frame cache size",
                        "Check data access patterns for locality",
                        "Consider data shuffling strategy",
                    ],
                    metrics={
                        "avg_cache_hit_rate": avg_cache_hit_rate,
                    }
                ))
        
        # Analyze worker performance variance
        if len(self._worker_metrics) > 1:
            worker_avg_times = {}
            for worker_id, worker_metrics in self._worker_metrics.items():
                if len(worker_metrics) >= 5:
                    worker_times = [m.batch_load_time for m in worker_metrics]
                    worker_avg_times[worker_id] = statistics.mean(worker_times)
            
            if len(worker_avg_times) > 1:
                times = list(worker_avg_times.values())
                time_variance = statistics.stdev(times) if len(times) > 1 else 0
                
                if time_variance > avg_load_time * 0.3:  # High variance between workers
                    bottlenecks.append(BottleneckAnalysis(
                        bottleneck_type="worker_load_imbalance",
                        severity="medium",
                        description=f"High variance in worker performance (std={time_variance:.4f}s)",
                        recommendations=[
                            "Check for uneven data distribution",
                            "Ensure all workers have similar system resources",
                            "Consider dynamic load balancing",
                            "Check for worker process issues",
                        ],
                        metrics={
                            "worker_time_variance": time_variance,
                            "worker_avg_times": worker_avg_times,
                        }
                    ))
        
        # Store and log bottlenecks
        self._bottlenecks = bottlenecks
        
        for bottleneck in bottlenecks:
            if bottleneck.severity in ["medium", "high"]:
                self._logger.warning(
                    f"Bottleneck detected: {bottleneck.bottleneck_type} "
                    f"({bottleneck.severity}): {bottleneck.description}"
                )
                for rec in bottleneck.recommendations[:2]:  # Log first 2 recommendations
                    self._logger.info(f"  Recommendation: {rec}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            if not self._metrics_history:
                return {}
            
            recent_metrics = list(self._metrics_history)
            
            # Calculate statistics
            load_times = [m.batch_load_time for m in recent_metrics]
            samples_per_sec = [m.samples_per_second for m in recent_metrics]
            video_decode_times = [m.video_decode_time for m in recent_metrics if m.video_decode_time > 0]
            cache_hit_rates = [m.cache_hit_rate for m in recent_metrics if m.cache_hit_rate >= 0]
            
            stats = {
                "total_batches": self._total_batches,
                "window_size": len(recent_metrics),
                "uptime_seconds": time.time() - self._start_time,
                "avg_batch_load_time": statistics.mean(load_times) if load_times else 0,
                "max_batch_load_time": max(load_times) if load_times else 0,
                "min_batch_load_time": min(load_times) if load_times else 0,
                "avg_samples_per_second": statistics.mean(samples_per_sec) if samples_per_sec else 0,
                "bottleneck_count": len(self._bottlenecks),
                "active_workers": len(self._worker_metrics),
            }
            
            if video_decode_times:
                stats.update({
                    "avg_video_decode_time": statistics.mean(video_decode_times),
                    "max_video_decode_time": max(video_decode_times),
                })
            
            if cache_hit_rates:
                stats.update({
                    "avg_cache_hit_rate": statistics.mean(cache_hit_rates),
                })
            
            return stats
    
    def get_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Get current bottleneck analysis."""
        with self._lock:
            return self._bottlenecks.copy()
    
    def get_worker_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get per-worker performance statistics."""
        with self._lock:
            worker_stats = {}
            
            for worker_id, metrics in self._worker_metrics.items():
                if not metrics:
                    continue
                
                load_times = [m.batch_load_time for m in metrics]
                samples_per_sec = [m.samples_per_second for m in metrics]
                
                worker_stats[worker_id] = {
                    "batch_count": len(metrics),
                    "avg_load_time": statistics.mean(load_times),
                    "max_load_time": max(load_times),
                    "avg_samples_per_second": statistics.mean(samples_per_sec),
                }
            
            return worker_stats
    
    def reset_stats(self):
        """Reset all statistics."""
        with self._lock:
            self._metrics_history.clear()
            self._worker_metrics.clear()
            self._batch_count = 0
            self._total_batches = 0
            self._start_time = time.time()
            self._bottlenecks.clear()
            self._last_analysis_time = time.time()
    
    def log_summary(self):
        """Log a comprehensive performance summary."""
        stats = self.get_current_stats()
        bottlenecks = self.get_bottlenecks()
        worker_stats = self.get_worker_stats()
        
        self._logger.info("=== Data Loading Performance Summary ===")
        self._logger.info(f"Total batches processed: {stats.get('total_batches', 0)}")
        self._logger.info(f"Uptime: {stats.get('uptime_seconds', 0):.1f} seconds")
        self._logger.info(f"Average batch load time: {stats.get('avg_batch_load_time', 0):.4f}s")
        self._logger.info(f"Average samples per second: {stats.get('avg_samples_per_second', 0):.1f}")
        self._logger.info(f"Active workers: {stats.get('active_workers', 0)}")
        
        if "avg_cache_hit_rate" in stats:
            self._logger.info(f"Average cache hit rate: {stats['avg_cache_hit_rate']:.2%}")
        
        if bottlenecks:
            self._logger.info(f"Active bottlenecks: {len(bottlenecks)}")
            for bottleneck in bottlenecks:
                self._logger.info(f"  - {bottleneck.bottleneck_type} ({bottleneck.severity})")
        
        if worker_stats:
            self._logger.info("Worker performance:")
            for worker_id, wstats in worker_stats.items():
                self._logger.info(
                    f"  Worker {worker_id}: {wstats['avg_load_time']:.4f}s avg, "
                    f"{wstats['avg_samples_per_second']:.1f} samples/s"
                )


# Global monitor instance (lazy initialization)
_global_monitor: Optional[DataLoadingMonitor] = None
_monitor_lock = threading.Lock()


def get_global_monitor() -> DataLoadingMonitor:
    """Get or create global data loading monitor instance."""
    global _global_monitor
    
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = DataLoadingMonitor()
        return _global_monitor


def reset_global_monitor():
    """Reset global monitor instance."""
    global _global_monitor
    
    with _monitor_lock:
        if _global_monitor is not None:
            _global_monitor.reset_stats()


def shutdown_global_monitor():
    """Shutdown global monitor instance."""
    global _global_monitor
    
    with _monitor_lock:
        if _global_monitor is not None:
            _global_monitor.log_summary()
            _global_monitor = None


# Cleanup on module exit
import atexit
atexit.register(shutdown_global_monitor)