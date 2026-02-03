#!/usr/bin/env python3
"""
Quick benchmark script to compare data loading performance with different settings.

Usage:
    # Test default optimized settings
    python benchmark_dataloader.py --config pi05_gr00t_local
    
    # Test with custom settings
    python benchmark_dataloader.py --config pi05_gr00t_local \
        --num-workers 4 --data.episode-cache-size 8
"""

import time
import argparse
from typing import Iterator
import numpy as np


def benchmark_dataloader(
    data_loader: Iterator,
    num_batches: int = 100,
    warmup_batches: int = 10,
) -> dict:
    """Benchmark data loader performance.
    
    Args:
        data_loader: Iterator yielding batches
        num_batches: Number of batches to measure (after warmup)
        warmup_batches: Number of warmup batches to skip
        
    Returns:
        Dictionary with timing statistics
    """
    print(f"Starting benchmark: {warmup_batches} warmup + {num_batches} measured batches")
    
    # Warmup phase
    print("Warming up...")
    for i, batch in enumerate(data_loader):
        if i >= warmup_batches:
            break
        if i % 5 == 0:
            print(f"  Warmup batch {i}/{warmup_batches}")
    
    # Measurement phase
    print(f"\nMeasuring {num_batches} batches...")
    batch_times = []
    
    for i in range(num_batches):
        start = time.time()
        batch = next(data_loader)
        elapsed_ms = (time.time() - start) * 1000
        batch_times.append(elapsed_ms)
        
        if i % 10 == 0:
            print(f"  Batch {i}/{num_batches}: {elapsed_ms:.1f}ms")
    
    # Compute statistics
    batch_times = np.array(batch_times)
    stats = {
        "mean": np.mean(batch_times),
        "median": np.median(batch_times),
        "std": np.std(batch_times),
        "min": np.min(batch_times),
        "max": np.max(batch_times),
        "p95": np.percentile(batch_times, 95),
        "p99": np.percentile(batch_times, 99),
    }
    
    return stats


def print_statistics(stats: dict):
    """Pretty print benchmark statistics."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Mean:   {stats['mean']:>8.1f} ms")
    print(f"Median: {stats['median']:>8.1f} ms")
    print(f"Std:    {stats['std']:>8.1f} ms")
    print(f"Min:    {stats['min']:>8.1f} ms")
    print(f"Max:    {stats['max']:>8.1f} ms")
    print(f"P95:    {stats['p95']:>8.1f} ms")
    print(f"P99:    {stats['p99']:>8.1f} ms")
    print("="*60)
    
    # Performance assessment
    if stats['mean'] < 20 and stats['p99'] < 50:
        print("✅ EXCELLENT: Data loading is not a bottleneck")
    elif stats['mean'] < 50 and stats['p99'] < 200:
        print("✓ GOOD: Acceptable performance, GPU should stay busy")
    elif stats['mean'] < 100:
        print("⚠ FAIR: May see some GPU starvation")
    else:
        print("❌ POOR: Data loading is a significant bottleneck")
        print("\nSuggestions:")
        print("  - Increase --num-workers (try 8 or 12)")
        print("  - Increase --data.episode-cache-size (try 16 or 32)")
        print("  - Check disk I/O with 'iotop'")
        print("  - See docs/performance_tuning.md for detailed guide")
    
    print("="*60 + "\n")


def main():
    """Run benchmark from command line."""
    parser = argparse.ArgumentParser(description="Benchmark OpenPI data loader performance")
    parser.add_argument("--config", type=str, default="pi05_gr00t_local",
                       help="Config name to benchmark")
    parser.add_argument("--num-batches", type=int, default=100,
                       help="Number of batches to measure")
    parser.add_argument("--warmup-batches", type=int, default=10,
                       help="Number of warmup batches")
    
    args, unknown = parser.parse_known_args()
    
    print(f"Benchmarking config: {args.config}")
    print(f"Extra args: {' '.join(unknown) if unknown else 'none'}\n")
    
    # Import here to avoid slow imports if just showing help
    import sys
    import os
    
    # Add openpi to path
    openpi_src = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, openpi_src)
    
    from openpi.training import config as _config
    from openpi.training import data_loader
    
    # Parse full config with any overrides
    full_args = [args.config] + unknown
    train_config = _config.parse_train_config(full_args)
    
    # Print configuration
    print("Configuration:")
    print(f"  num_workers: {train_config.num_workers}")
    print(f"  batch_size: {train_config.batch_size}")
    if hasattr(train_config.data, 'episode_cache_size'):
        print(f"  episode_cache_size: {train_config.data.episode_cache_size}")
    if hasattr(train_config.data, 'dataset_path'):
        print(f"  dataset_path: {train_config.data.dataset_path}")
    print()
    
    # Create data loader
    print("Creating data loader...")
    loader = data_loader.create_data_loader(
        train_config,
        shuffle=True,
        num_batches=args.warmup_batches + args.num_batches,
    )
    
    # Run benchmark
    stats = benchmark_dataloader(
        iter(loader),
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
    )
    
    # Print results
    print_statistics(stats)


if __name__ == "__main__":
    main()
