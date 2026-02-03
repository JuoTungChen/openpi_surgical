"""Advanced bottleneck detection algorithms for OpenPI training pipeline."""

import dataclasses
import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from openpi.training.gpu_utilization_monitor import BottleneckType, PerformanceMetrics


@dataclasses.dataclass
class BottleneckAlert:
    """Represents a bottleneck alert with severity and recommendations."""
    bottleneck_type: BottleneckType
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0
    message: str
    detected_at: float
    metrics_evidence: Dict[str, float]
    recommendations: List[str]


class AdvancedBottleneckDetector:
    """Advanced bottleneck detection using statistical analysis and thresholds."""
    
    def __init__(
        self,
        analysis_window_size: int = 30,
        alert_cooldown_seconds: float = 60.0,
        confidence_threshold: float = 0.7,
    ):
        """Initialize advanced bottleneck detector.
        
        Args:
            analysis_window_size: Number of recent metrics to analyze
            alert_cooldown_seconds: Minimum time between alerts of same type
            confidence_threshold: Minimum confidence to trigger an alert
        """
        self.analysis_window_size = analysis_window_size
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.confidence_threshold = confidence_threshold
        
        # Alert tracking
        self.recent_alerts: Dict[BottleneckType, float] = {}
        self.alert_history: deque[BottleneckAlert] = deque(maxlen=100)
        
        # Performance baselines (learned over time)
        self.baseline_utilization: Optional[float] = None
        self.baseline_memory_usage: Optional[float] = None
        self.baseline_variability: Optional[float] = None
        
        # Thresholds for different severity levels
        self.thresholds = {
            "data_loading": {
                "low": {"utilization": 0.6, "duration_ratio": 0.1},
                "medium": {"utilization": 0.4, "duration_ratio": 0.2},
                "high": {"utilization": 0.3, "duration_ratio": 0.3},
                "critical": {"utilization": 0.2, "duration_ratio": 0.5},
            },
            "memory_pressure": {
                "low": {"usage": 0.85, "growth_rate": 0.01},
                "medium": {"usage": 0.90, "growth_rate": 0.02},
                "high": {"usage": 0.95, "growth_rate": 0.05},
                "critical": {"usage": 0.98, "growth_rate": 0.1},
            },
            "computation_inefficiency": {
                "low": {"variability": 0.2, "low_util_ratio": 0.1},
                "medium": {"variability": 0.3, "low_util_ratio": 0.2},
                "high": {"variability": 0.4, "low_util_ratio": 0.3},
                "critical": {"variability": 0.5, "low_util_ratio": 0.5},
            },
        }
    
    def analyze_metrics(self, metrics_history: List[PerformanceMetrics]) -> List[BottleneckAlert]:
        """Analyze metrics history and detect bottlenecks."""
        if len(metrics_history) < self.analysis_window_size:
            return []
        
        # Use the most recent metrics for analysis
        recent_metrics = metrics_history[-self.analysis_window_size:]
        current_time = time.time()
        
        alerts = []
        
        # Update baselines if needed
        self._update_baselines(recent_metrics)
        
        # Detect different types of bottlenecks
        alerts.extend(self._detect_data_loading_bottlenecks(recent_metrics, current_time))
        alerts.extend(self._detect_memory_bottlenecks(recent_metrics, current_time))
        alerts.extend(self._detect_computation_bottlenecks(recent_metrics, current_time))
        alerts.extend(self._detect_gradient_sync_bottlenecks(recent_metrics, current_time))
        
        # Filter alerts based on confidence and cooldown
        filtered_alerts = self._filter_alerts(alerts, current_time)
        
        # Store alerts in history
        for alert in filtered_alerts:
            self.alert_history.append(alert)
            self.recent_alerts[alert.bottleneck_type] = current_time
        
        return filtered_alerts
    
    def _update_baselines(self, metrics: List[PerformanceMetrics]) -> None:
        """Update performance baselines based on recent metrics."""
        utilizations = [m.gpu_utilization for m in metrics]
        memory_usages = [m.gpu_memory_usage / max(m.gpu_memory_total, 1) for m in metrics]
        
        # Update utilization baseline (moving average)
        avg_utilization = np.mean(utilizations)
        if self.baseline_utilization is None:
            self.baseline_utilization = avg_utilization
        else:
            self.baseline_utilization = 0.9 * self.baseline_utilization + 0.1 * avg_utilization
        
        # Update memory usage baseline
        avg_memory = np.mean(memory_usages)
        if self.baseline_memory_usage is None:
            self.baseline_memory_usage = avg_memory
        else:
            self.baseline_memory_usage = 0.9 * self.baseline_memory_usage + 0.1 * avg_memory
        
        # Update variability baseline
        variability = np.std(utilizations) if len(utilizations) > 1 else 0.0
        if self.baseline_variability is None:
            self.baseline_variability = variability
        else:
            self.baseline_variability = 0.9 * self.baseline_variability + 0.1 * variability
    
    def _detect_data_loading_bottlenecks(
        self, metrics: List[PerformanceMetrics], current_time: float
    ) -> List[BottleneckAlert]:
        """Detect data loading bottlenecks based on GPU utilization patterns."""
        alerts = []
        
        utilizations = [m.gpu_utilization for m in metrics]
        avg_utilization = np.mean(utilizations)
        
        # Calculate time spent with low utilization
        low_util_count = sum(1 for u in utilizations if u < 0.5)
        low_util_ratio = low_util_count / len(utilizations)
        
        # Detect data loading times if available
        data_loading_times = [m.data_loading_time for m in metrics if m.data_loading_time is not None]
        avg_data_loading_time = np.mean(data_loading_times) if data_loading_times else None
        
        # Determine severity based on thresholds
        severity = None
        confidence = 0.0
        
        for sev_level in ["critical", "high", "medium", "low"]:
            thresholds = self.thresholds["data_loading"][sev_level]
            if (avg_utilization < thresholds["utilization"] and 
                low_util_ratio > thresholds["duration_ratio"]):
                severity = sev_level
                confidence = min(1.0, (thresholds["utilization"] - avg_utilization) / 0.3 + 
                               low_util_ratio)
                break
        
        if severity and confidence >= self.confidence_threshold:
            evidence = {
                "avg_utilization": avg_utilization,
                "low_utilization_ratio": low_util_ratio,
                "baseline_utilization": self.baseline_utilization or 0.0,
            }
            
            if avg_data_loading_time:
                evidence["avg_data_loading_time"] = avg_data_loading_time
            
            recommendations = self._get_data_loading_recommendations(severity, evidence)
            
            alerts.append(BottleneckAlert(
                bottleneck_type=BottleneckType.DATA_LOADING,
                severity=severity,
                confidence=confidence,
                message=f"Data loading bottleneck detected: {avg_utilization:.1%} avg GPU utilization, "
                       f"{low_util_ratio:.1%} time with low utilization",
                detected_at=current_time,
                metrics_evidence=evidence,
                recommendations=recommendations,
            ))
        
        return alerts
    
    def _detect_memory_bottlenecks(
        self, metrics: List[PerformanceMetrics], current_time: float
    ) -> List[BottleneckAlert]:
        """Detect memory-related bottlenecks."""
        alerts = []
        
        memory_usages = [m.gpu_memory_usage / max(m.gpu_memory_total, 1) for m in metrics]
        avg_memory_usage = np.mean(memory_usages)
        max_memory_usage = np.max(memory_usages)
        
        # Calculate memory growth rate
        if len(memory_usages) > 5:
            recent_usage = memory_usages[-5:]
            older_usage = memory_usages[-10:-5] if len(memory_usages) >= 10 else memory_usages[:-5]
            growth_rate = (np.mean(recent_usage) - np.mean(older_usage)) if older_usage else 0.0
        else:
            growth_rate = 0.0
        
        # Determine severity
        severity = None
        confidence = 0.0
        
        for sev_level in ["critical", "high", "medium", "low"]:
            thresholds = self.thresholds["memory_pressure"][sev_level]
            if (avg_memory_usage > thresholds["usage"] or 
                growth_rate > thresholds["growth_rate"]):
                severity = sev_level
                confidence = min(1.0, max(
                    (avg_memory_usage - thresholds["usage"]) / 0.1,
                    growth_rate / thresholds["growth_rate"]
                ))
                break
        
        if severity and confidence >= self.confidence_threshold:
            evidence = {
                "avg_memory_usage": avg_memory_usage,
                "max_memory_usage": max_memory_usage,
                "memory_growth_rate": growth_rate,
                "baseline_memory_usage": self.baseline_memory_usage or 0.0,
            }
            
            recommendations = self._get_memory_recommendations(severity, evidence)
            
            alerts.append(BottleneckAlert(
                bottleneck_type=BottleneckType.MEMORY_TRANSFER,
                severity=severity,
                confidence=confidence,
                message=f"Memory pressure detected: {avg_memory_usage:.1%} avg usage, "
                       f"{growth_rate:.3f} growth rate",
                detected_at=current_time,
                metrics_evidence=evidence,
                recommendations=recommendations,
            ))
        
        return alerts
    
    def _detect_computation_bottlenecks(
        self, metrics: List[PerformanceMetrics], current_time: float
    ) -> List[BottleneckAlert]:
        """Detect computation inefficiency bottlenecks."""
        alerts = []
        
        utilizations = [m.gpu_utilization for m in metrics]
        variability = np.std(utilizations) if len(utilizations) > 1 else 0.0
        
        # Calculate ratio of time with low utilization
        low_util_count = sum(1 for u in utilizations if u < 0.3)
        low_util_ratio = low_util_count / len(utilizations)
        
        # Determine severity
        severity = None
        confidence = 0.0
        
        for sev_level in ["critical", "high", "medium", "low"]:
            thresholds = self.thresholds["computation_inefficiency"][sev_level]
            if (variability > thresholds["variability"] and 
                low_util_ratio > thresholds["low_util_ratio"]):
                severity = sev_level
                confidence = min(1.0, variability / 0.5 + low_util_ratio)
                break
        
        if severity and confidence >= self.confidence_threshold:
            evidence = {
                "utilization_variability": variability,
                "low_utilization_ratio": low_util_ratio,
                "baseline_variability": self.baseline_variability or 0.0,
            }
            
            recommendations = self._get_computation_recommendations(severity, evidence)
            
            alerts.append(BottleneckAlert(
                bottleneck_type=BottleneckType.GPU_COMPUTATION,
                severity=severity,
                confidence=confidence,
                message=f"Computation inefficiency detected: {variability:.2f} utilization variability, "
                       f"{low_util_ratio:.1%} time with very low utilization",
                detected_at=current_time,
                metrics_evidence=evidence,
                recommendations=recommendations,
            ))
        
        return alerts
    
    def _detect_gradient_sync_bottlenecks(
        self, metrics: List[PerformanceMetrics], current_time: float
    ) -> List[BottleneckAlert]:
        """Detect gradient synchronization bottlenecks."""
        alerts = []
        
        # Look for gradient sync times if available
        sync_times = [m.gradient_sync_time for m in metrics if m.gradient_sync_time is not None]
        
        if not sync_times:
            return alerts  # No gradient sync data available
        
        avg_sync_time = np.mean(sync_times)
        max_sync_time = np.max(sync_times)
        
        # Thresholds for gradient sync (in seconds)
        sync_thresholds = {
            "low": 0.1,
            "medium": 0.2,
            "high": 0.5,
            "critical": 1.0,
        }
        
        severity = None
        confidence = 0.0
        
        for sev_level in ["critical", "high", "medium", "low"]:
            if avg_sync_time > sync_thresholds[sev_level]:
                severity = sev_level
                confidence = min(1.0, avg_sync_time / sync_thresholds[sev_level])
                break
        
        if severity and confidence >= self.confidence_threshold:
            evidence = {
                "avg_gradient_sync_time": avg_sync_time,
                "max_gradient_sync_time": max_sync_time,
            }
            
            recommendations = [
                "Consider using asynchronous gradient synchronization",
                "Check network bandwidth between GPUs",
                "Optimize gradient compression settings",
                "Reduce gradient synchronization frequency",
                "Use gradient accumulation to reduce sync overhead",
            ]
            
            alerts.append(BottleneckAlert(
                bottleneck_type=BottleneckType.GRADIENT_SYNC,
                severity=severity,
                confidence=confidence,
                message=f"Gradient synchronization bottleneck: {avg_sync_time:.3f}s avg sync time",
                detected_at=current_time,
                metrics_evidence=evidence,
                recommendations=recommendations,
            ))
        
        return alerts
    
    def _filter_alerts(self, alerts: List[BottleneckAlert], current_time: float) -> List[BottleneckAlert]:
        """Filter alerts based on confidence and cooldown periods."""
        filtered = []
        
        for alert in alerts:
            # Check confidence threshold
            if alert.confidence < self.confidence_threshold:
                continue
            
            # Check cooldown period
            last_alert_time = self.recent_alerts.get(alert.bottleneck_type, 0)
            if current_time - last_alert_time < self.alert_cooldown_seconds:
                continue
            
            filtered.append(alert)
        
        return filtered
    
    def _get_data_loading_recommendations(self, severity: str, evidence: Dict[str, float]) -> List[str]:
        """Get recommendations for data loading bottlenecks."""
        base_recommendations = [
            "Increase num_workers in DataLoader (try 4-8 workers)",
            "Enable prefetch_factor=4 or higher in DataLoader",
            "Use persistent_workers=True to avoid worker restart overhead",
            "Enable pin_memory=True for faster CPU-GPU transfers",
        ]
        
        if severity in ["high", "critical"]:
            base_recommendations.extend([
                "Consider async video decoding for GR00T datasets",
                "Implement data caching to reduce I/O overhead",
                "Profile data preprocessing pipeline for bottlenecks",
                "Use faster storage (SSD) for dataset if possible",
            ])
        
        if evidence.get("avg_data_loading_time", 0) > 0.1:
            base_recommendations.append("Data loading time is high - optimize preprocessing")
        
        return base_recommendations
    
    def _get_memory_recommendations(self, severity: str, evidence: Dict[str, float]) -> List[str]:
        """Get recommendations for memory bottlenecks."""
        base_recommendations = [
            "Reduce batch size to free up GPU memory",
            "Enable gradient accumulation to maintain effective batch size",
            "Use mixed precision training (bfloat16) to reduce memory usage",
        ]
        
        if severity in ["high", "critical"]:
            base_recommendations.extend([
                "Enable FSDP model sharding across multiple GPUs",
                "Use gradient checkpointing to trade compute for memory",
                "Clear unnecessary variables and call jax.clear_caches()",
                "Consider using CPU offloading for optimizer states",
            ])
        
        if evidence.get("memory_growth_rate", 0) > 0.02:
            base_recommendations.append("Memory leak detected - check for accumulating variables")
        
        return base_recommendations
    
    def _get_computation_recommendations(self, severity: str, evidence: Dict[str, float]) -> List[str]:
        """Get recommendations for computation bottlenecks."""
        base_recommendations = [
            "Warm up JIT compilation cache with sample data",
            "Use static shapes to avoid recompilation",
            "Profile JAX compilation times and optimize hot paths",
        ]
        
        if severity in ["high", "critical"]:
            base_recommendations.extend([
                "Check for gradient synchronization bottlenecks",
                "Optimize model architecture for better GPU utilization",
                "Consider using compilation cache persistence",
                "Profile memory transfer patterns",
            ])
        
        if evidence.get("utilization_variability", 0) > 0.4:
            base_recommendations.append("High variability suggests compilation issues")
        
        return base_recommendations
    
    def get_alert_summary(self) -> Dict[str, any]:
        """Get a summary of recent alerts."""
        if not self.alert_history:
            return {"total_alerts": 0, "recent_alerts": 0}
        
        current_time = time.time()
        recent_alerts = [a for a in self.alert_history if current_time - a.detected_at < 300]  # Last 5 minutes
        
        # Count by severity and type
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1
            type_counts[alert.bottleneck_type.value] += 1
        
        return {
            "total_alerts": len(self.alert_history),
            "recent_alerts": len(recent_alerts),
            "recent_by_severity": dict(severity_counts),
            "recent_by_type": dict(type_counts),
            "last_alert_time": max(a.detected_at for a in self.alert_history) if self.alert_history else None,
        }