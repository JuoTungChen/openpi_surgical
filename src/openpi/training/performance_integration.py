"""Integration module for GPU utilization monitoring in OpenPI training pipeline."""

import dataclasses
import logging
import time
from typing import Any, Dict, Optional

import wandb

from openpi.training.bottleneck_detector import AdvancedBottleneckDetector, BottleneckAlert
from openpi.training.gpu_utilization_monitor import GPUUtilizationMonitor, PerformanceMetrics


@dataclasses.dataclass
class PerformanceIntegrationConfig:
    """Configuration for performance monitoring integration."""
    # GPU monitoring settings
    target_gpu_utilization: float = 0.85
    monitoring_interval: float = 1.0
    metrics_window_size: int = 60
    
    # Bottleneck detection settings
    bottleneck_analysis_window: int = 30
    alert_cooldown_seconds: float = 60.0
    confidence_threshold: float = 0.7
    
    # Logging and reporting settings
    performance_log_interval: int = 100  # Log every N steps
    wandb_log_interval: int = 10  # Log to wandb every N steps
    detailed_report_interval: int = 1000  # Detailed report every N steps
    
    # Enable/disable features
    enable_gpu_monitoring: bool = True
    enable_bottleneck_detection: bool = True
    enable_wandb_logging: bool = True
    enable_automatic_suggestions: bool = True


class PerformanceIntegrator:
    """Integrates GPU monitoring and bottleneck detection into training pipeline."""
    
    def __init__(self, config: PerformanceIntegrationConfig):
        """Initialize performance integrator.
        
        Args:
            config: Configuration for performance monitoring
        """
        self.config = config
        
        # Initialize monitoring components
        self.gpu_monitor: Optional[GPUUtilizationMonitor] = None
        self.bottleneck_detector: Optional[AdvancedBottleneckDetector] = None
        
        if config.enable_gpu_monitoring:
            self.gpu_monitor = GPUUtilizationMonitor(
                target_utilization=config.target_gpu_utilization,
                monitoring_interval=config.monitoring_interval,
                metrics_window_size=config.metrics_window_size,
            )
        
        if config.enable_bottleneck_detection:
            self.bottleneck_detector = AdvancedBottleneckDetector(
                analysis_window_size=config.bottleneck_analysis_window,
                alert_cooldown_seconds=config.alert_cooldown_seconds,
                confidence_threshold=config.confidence_threshold,
            )
        
        # State tracking
        self.training_start_time: Optional[float] = None
        self.last_performance_log: int = 0
        self.last_wandb_log: int = 0
        self.last_detailed_report: int = 0
        
        logging.info("Initialized performance monitoring integration")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring()
            logging.info("Started GPU utilization monitoring")
        
        self.training_start_time = time.time()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self.gpu_monitor:
            self.gpu_monitor.stop_monitoring()
            logging.info("Stopped GPU utilization monitoring")
    
    def on_training_step_start(self, step: int) -> None:
        """Called at the start of each training step."""
        if self.gpu_monitor:
            self.gpu_monitor.mark_step_start()
    
    def on_data_loading_start(self) -> None:
        """Called when data loading starts."""
        if self.gpu_monitor:
            self.gpu_monitor.mark_data_loading_start()
    
    def on_data_loading_end(self) -> None:
        """Called when data loading ends."""
        if self.gpu_monitor:
            self.gpu_monitor.mark_data_loading_end()
    
    def on_computation_start(self) -> None:
        """Called when GPU computation starts."""
        if self.gpu_monitor:
            self.gpu_monitor.mark_computation_start()
    
    def on_computation_end(self) -> None:
        """Called when GPU computation ends."""
        if self.gpu_monitor:
            self.gpu_monitor.mark_computation_end()
    
    def on_training_step_end(self, step: int, step_info: Dict[str, Any]) -> None:
        """Called at the end of each training step.
        
        Args:
            step: Current training step
            step_info: Dictionary containing step metrics (loss, grad_norm, etc.)
        """
        # Log performance metrics at specified intervals
        if (step - self.last_performance_log) >= self.config.performance_log_interval:
            self._log_performance_metrics(step)
            self.last_performance_log = step
        
        # Log to wandb at specified intervals
        if (self.config.enable_wandb_logging and 
            (step - self.last_wandb_log) >= self.config.wandb_log_interval):
            self._log_to_wandb(step, step_info)
            self.last_wandb_log = step
        
        # Generate detailed reports at specified intervals
        if (step - self.last_detailed_report) >= self.config.detailed_report_interval:
            self._generate_detailed_report(step)
            self.last_detailed_report = step
        
        # Check for bottlenecks and alerts
        if self.config.enable_bottleneck_detection:
            self._check_bottlenecks(step)
    
    def _log_performance_metrics(self, step: int) -> None:
        """Log performance metrics to the logger."""
        if not self.gpu_monitor:
            return
        
        self.gpu_monitor.log_performance_report(step)
    
    def _log_to_wandb(self, step: int, step_info: Dict[str, Any]) -> None:
        """Log performance metrics to wandb."""
        if not self.gpu_monitor:
            return
        
        try:
            # Get current performance metrics
            summary = self.gpu_monitor.get_performance_summary()
            memory_stats = self.gpu_monitor.get_memory_usage_stats()
            
            # Prepare wandb metrics
            wandb_metrics = {
                "performance/gpu_utilization_current": summary.get("current_utilization", 0.0),
                "performance/gpu_utilization_avg_1min": summary.get("average_utilization_1min", 0.0),
                "performance/gpu_utilization_target": summary.get("target_utilization", 0.0),
                "performance/low_utilization_percentage": summary.get("low_utilization_percentage", 0.0),
                "performance/gpu_memory_usage_fraction": memory_stats.get("usage_fraction", 0.0),
                "performance/gpu_memory_usage_gb": memory_stats.get("usage_bytes", 0) / (1024**3),
                "performance/gpu_memory_total_gb": memory_stats.get("total_bytes", 1) / (1024**3),
                "performance/device_count": summary.get("device_count", 1),
                "performance/metrics_collected": summary.get("metrics_collected", 0),
            }
            
            # Add bottleneck information if available
            if self.bottleneck_detector:
                alert_summary = self.bottleneck_detector.get_alert_summary()
                wandb_metrics.update({
                    "performance/total_alerts": alert_summary.get("total_alerts", 0),
                    "performance/recent_alerts": alert_summary.get("recent_alerts", 0),
                })
                
                # Add severity counts
                severity_counts = alert_summary.get("recent_by_severity", {})
                for severity, count in severity_counts.items():
                    wandb_metrics[f"performance/alerts_{severity}"] = count
            
            # Add training efficiency metrics
            if self.training_start_time:
                training_duration = time.time() - self.training_start_time
                steps_per_second = step / max(training_duration, 1)
                wandb_metrics["performance/steps_per_second"] = steps_per_second
                wandb_metrics["performance/training_duration_minutes"] = training_duration / 60
            
            # Log to wandb
            wandb.log(wandb_metrics, step=step)
            
        except Exception as e:
            logging.warning(f"Failed to log performance metrics to wandb: {e}")
    
    def _check_bottlenecks(self, step: int) -> None:
        """Check for bottlenecks and log alerts."""
        if not (self.gpu_monitor and self.bottleneck_detector):
            return
        
        try:
            # Get metrics history from GPU monitor
            with self.gpu_monitor._lock:
                metrics_history = list(self.gpu_monitor.metrics_history)
            
            if not metrics_history:
                return
            
            # Analyze for bottlenecks
            alerts = self.bottleneck_detector.analyze_metrics(metrics_history)
            
            # Log alerts
            for alert in alerts:
                self._log_bottleneck_alert(step, alert)
                
                # Log suggestions if enabled
                if self.config.enable_automatic_suggestions:
                    self._log_optimization_suggestions(step, alert)
        
        except Exception as e:
            logging.warning(f"Failed to check bottlenecks: {e}")
    
    def _log_bottleneck_alert(self, step: int, alert: BottleneckAlert) -> None:
        """Log a bottleneck alert."""
        severity_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🚨",
            "critical": "🔥"
        }
        
        emoji = severity_emoji.get(alert.severity, "⚠️")
        
        log_func = logging.info
        if alert.severity in ["high", "critical"]:
            log_func = logging.warning
        
        log_func(
            f"Step {step}: {emoji} {alert.severity.upper()} bottleneck detected - "
            f"{alert.bottleneck_type.value}: {alert.message} "
            f"(confidence: {alert.confidence:.1%})"
        )
        
        # Log to wandb as well
        try:
            wandb.log({
                f"alerts/{alert.bottleneck_type.value}_{alert.severity}": 1,
                f"alerts/confidence_{alert.bottleneck_type.value}": alert.confidence,
            }, step=step)
        except Exception as e:
            logging.debug(f"Failed to log alert to wandb: {e}")
    
    def _log_optimization_suggestions(self, step: int, alert: BottleneckAlert) -> None:
        """Log optimization suggestions for an alert."""
        if not alert.recommendations:
            return
        
        logging.info(f"Step {step}: Optimization suggestions for {alert.bottleneck_type.value}:")
        for i, suggestion in enumerate(alert.recommendations[:3], 1):  # Limit to top 3
            logging.info(f"  {i}. {suggestion}")
    
    def _generate_detailed_report(self, step: int) -> None:
        """Generate a detailed performance report."""
        if not self.gpu_monitor:
            return
        
        try:
            summary = self.gpu_monitor.get_performance_summary()
            
            logging.info(f"=== Performance Report - Step {step} ===")
            logging.info(f"GPU Utilization: {summary.get('current_utilization', 0):.1%} current, "
                        f"{summary.get('average_utilization_1min', 0):.1%} avg (target: "
                        f"{summary.get('target_utilization', 0):.1%})")
            
            memory_gb = summary.get('memory_usage_gb', 0)
            memory_total_gb = summary.get('memory_total_gb', 1)
            logging.info(f"GPU Memory: {memory_gb:.1f}GB / {memory_total_gb:.1f}GB "
                        f"({summary.get('memory_usage_fraction', 0):.1%})")
            
            low_util_pct = summary.get('low_utilization_percentage', 0)
            if low_util_pct > 5:
                logging.info(f"⚠️  Low utilization warning: {low_util_pct:.1f}% of time below 50%")
            
            # Bottleneck summary
            if self.bottleneck_detector:
                alert_summary = self.bottleneck_detector.get_alert_summary()
                recent_alerts = alert_summary.get('recent_alerts', 0)
                if recent_alerts > 0:
                    logging.info(f"Recent alerts (5min): {recent_alerts}")
                    
                    by_type = alert_summary.get('recent_by_type', {})
                    if by_type:
                        types_str = ", ".join(f"{k}: {v}" for k, v in by_type.items())
                        logging.info(f"Alert breakdown: {types_str}")
            
            # Training efficiency
            if self.training_start_time:
                duration = time.time() - self.training_start_time
                steps_per_sec = step / max(duration, 1)
                logging.info(f"Training efficiency: {steps_per_sec:.2f} steps/sec, "
                           f"{duration/60:.1f} minutes elapsed")
            
            logging.info("=" * 50)
            
        except Exception as e:
            logging.warning(f"Failed to generate detailed report: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        summary = {"monitoring_enabled": False}
        
        if self.gpu_monitor:
            summary.update(self.gpu_monitor.get_performance_summary())
            summary["monitoring_enabled"] = True
        
        if self.bottleneck_detector:
            summary.update(self.bottleneck_detector.get_alert_summary())
        
        if self.training_start_time:
            summary["training_duration_seconds"] = time.time() - self.training_start_time
        
        return summary