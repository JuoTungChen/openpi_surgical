#!/usr/bin/env python3
"""Test script for GPU utilization monitoring system."""

import logging
import time
from typing import Dict, Any

import jax
import jax.numpy as jnp

from openpi.training.gpu_utilization_monitor import GPUUtilizationMonitor
from openpi.training.bottleneck_detector import AdvancedBottleneckDetector
from openpi.training.performance_integration import PerformanceIntegrator, PerformanceIntegrationConfig


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def simulate_training_workload():
    """Simulate a training workload to test monitoring."""
    logging.info("Simulating training workload...")
    
    # Create some dummy computation to use GPU
    key = jax.random.key(42)
    
    for i in range(10):
        # Simulate data loading delay
        time.sleep(0.1)
        
        # Simulate GPU computation
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (1000, 1000))
        y = jnp.dot(x, x.T)
        result = jnp.sum(y)
        
        # Block until computation is done
        jax.block_until_ready(result)
        
        logging.info(f"Step {i}: computation result = {result:.2f}")
        time.sleep(0.2)


def test_gpu_monitor():
    """Test the GPU utilization monitor."""
    logging.info("Testing GPU utilization monitor...")
    
    monitor = GPUUtilizationMonitor(
        target_utilization=0.8,
        monitoring_interval=0.5,
        metrics_window_size=20
    )
    
    try:
        monitor.start_monitoring()
        
        # Let it collect some baseline metrics
        time.sleep(2)
        
        # Simulate some training steps
        for step in range(5):
            monitor.mark_step_start()
            monitor.mark_data_loading_start()
            time.sleep(0.1)  # Simulate data loading
            monitor.mark_data_loading_end()
            
            monitor.mark_computation_start()
            simulate_training_workload()
            monitor.mark_computation_end()
            
            # Log performance report
            monitor.log_performance_report(step)
        
        # Get final summary
        summary = monitor.get_performance_summary()
        logging.info("GPU Monitor Summary:")
        for key, value in summary.items():
            logging.info(f"  {key}: {value}")
        
        # Test bottleneck detection
        bottlenecks = monitor.detect_bottlenecks()
        logging.info(f"Detected bottlenecks: {[b.value for b in bottlenecks]}")
        
        # Test optimization suggestions
        suggestions = monitor.suggest_optimizations()
        logging.info(f"Optimization suggestions: {len(suggestions)}")
        for suggestion in suggestions:
            logging.info(f"  - {suggestion.message}")
    
    finally:
        monitor.stop_monitoring()


def test_bottleneck_detector():
    """Test the advanced bottleneck detector."""
    logging.info("Testing bottleneck detector...")
    
    detector = AdvancedBottleneckDetector(
        analysis_window_size=10,
        alert_cooldown_seconds=5.0,
        confidence_threshold=0.5
    )
    
    # Create some fake metrics to test detection
    from openpi.training.gpu_utilization_monitor import PerformanceMetrics
    
    metrics = []
    current_time = time.time()
    
    # Simulate low GPU utilization (data loading bottleneck)
    for i in range(15):
        metrics.append(PerformanceMetrics(
            timestamp=current_time + i,
            gpu_utilization=0.3,  # Low utilization
            gpu_memory_usage=1024 * 1024 * 1024,  # 1GB
            gpu_memory_total=8 * 1024 * 1024 * 1024,  # 8GB
            data_loading_time=0.2,  # High data loading time
        ))
    
    alerts = detector.analyze_metrics(metrics)
    logging.info(f"Generated {len(alerts)} alerts")
    
    for alert in alerts:
        logging.info(f"Alert: {alert.severity} {alert.bottleneck_type.value} - {alert.message}")
        logging.info(f"  Confidence: {alert.confidence:.1%}")
        logging.info(f"  Recommendations: {len(alert.recommendations)}")
        for rec in alert.recommendations[:3]:  # Show first 3
            logging.info(f"    - {rec}")


def test_performance_integration():
    """Test the performance integration system."""
    logging.info("Testing performance integration...")
    
    config = PerformanceIntegrationConfig(
        target_gpu_utilization=0.8,
        monitoring_interval=0.5,
        enable_gpu_monitoring=True,
        enable_bottleneck_detection=True,
        enable_wandb_logging=False,  # Disable wandb for testing
        performance_log_interval=2,
    )
    
    integrator = PerformanceIntegrator(config)
    
    try:
        integrator.start_monitoring()
        
        # Simulate training steps
        for step in range(5):
            integrator.on_training_step_start(step)
            
            integrator.on_data_loading_start()
            time.sleep(0.1)  # Simulate data loading
            integrator.on_data_loading_end()
            
            integrator.on_computation_start()
            # Simulate some computation
            key = jax.random.key(step)
            x = jax.random.normal(key, (500, 500))
            result = jnp.sum(x @ x.T)
            jax.block_until_ready(result)
            integrator.on_computation_end()
            
            # Simulate step info
            step_info = {"loss": 0.5 - step * 0.05, "grad_norm": 1.0}
            integrator.on_training_step_end(step, step_info)
            
            time.sleep(0.5)
        
        # Get final summary
        summary = integrator.get_performance_summary()
        logging.info("Performance Integration Summary:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {key}: {value}")
    
    finally:
        integrator.stop_monitoring()


def main():
    """Run all tests."""
    setup_logging()
    
    logging.info("Starting GPU monitoring system tests...")
    logging.info(f"JAX devices: {jax.devices()}")
    
    try:
        test_gpu_monitor()
        logging.info("✓ GPU monitor test completed")
        
        test_bottleneck_detector()
        logging.info("✓ Bottleneck detector test completed")
        
        test_performance_integration()
        logging.info("✓ Performance integration test completed")
        
        logging.info("All tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()