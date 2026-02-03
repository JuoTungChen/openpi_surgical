"""
Intelligent episode caching system for GR00T dataset optimization.

This module provides memory-aware and distributed caching strategies to improve
data loading performance and reduce I/O overhead.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import psutil
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import weakref

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("openpi requires pandas for data loading.") from e


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    max_memory_bytes: int = 0
    cache_size: int = 0
    max_cache_size: int = 0
    hit_rate: float = 0.0
    memory_utilization: float = 0.0


@dataclass
class EpisodeCacheEntry:
    """Entry in the episode cache."""
    episode_id: int
    data: Any
    size_bytes: int
    access_count: int
    last_access_time: float
    creation_time: float
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()


class MemoryAwareLRUCache:
    """
    Memory-aware LRU cache that evicts items based on both usage and memory pressure.
    
    This cache implements an intelligent eviction policy that considers:
    - LRU (Least Recently Used) ordering
    - Memory usage and available system memory
    - Access frequency
    - Item size
    """
    
    def __init__(
        self,
        max_size: int = 128,
        max_memory_mb: int = 1024,
        memory_pressure_threshold: float = 0.8,
        enable_memory_monitoring: bool = True,
        cleanup_interval: float = 30.0,
    ):
        """
        Initialize memory-aware LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            max_memory_mb: Maximum memory usage in MB
            memory_pressure_threshold: Memory pressure threshold (0.0-1.0)
            enable_memory_monitoring: Whether to monitor system memory
            cleanup_interval: Interval for background cleanup in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_pressure_threshold = memory_pressure_threshold
        self.enable_memory_monitoring = enable_memory_monitoring
        self.cleanup_interval = cleanup_interval
        
        # Cache storage
        self._cache: OrderedDict[Any, EpisodeCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = CacheStats(max_memory_bytes=self.max_memory_bytes, max_cache_size=max_size)
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        if enable_memory_monitoring:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(
                target=self._background_cleanup,
                daemon=True,
                name="EpisodeCacheCleanup"
            )
            self._cleanup_thread.start()
    
    def _background_cleanup(self):
        """Background cleanup thread function."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_if_needed()
            except Exception as e:
                warnings.warn(
                    f"Background cache cleanup failed: {e}",
                    UserWarning,
                    stacklevel=2,
                )
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data object."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, dict):
                total = 0
                for key, value in data.items():
                    total += self._estimate_size(key) + self._estimate_size(value)
                return total
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            else:
                # Fallback: use sys.getsizeof with some overhead estimation
                import sys
                return sys.getsizeof(data) * 2  # Add overhead factor
        except Exception:
            # Conservative estimate if size calculation fails
            return 1024 * 1024  # 1MB default
    
    def _get_memory_pressure(self) -> float:
        """Get current system memory pressure (0.0-1.0)."""
        if not self.enable_memory_monitoring:
            return 0.0
        
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except Exception:
            return 0.0
    
    def _should_evict_for_memory(self) -> bool:
        """Check if we should evict items due to memory pressure."""
        if not self.enable_memory_monitoring:
            return False
        
        # Check cache memory usage
        if self._stats.memory_usage_bytes > self.max_memory_bytes:
            return True
        
        # Check system memory pressure
        memory_pressure = self._get_memory_pressure()
        return memory_pressure > self.memory_pressure_threshold
    
    def _evict_items(self, target_size: Optional[int] = None, target_memory: Optional[int] = None):
        """Evict items from cache based on LRU and memory considerations."""
        if not self._cache:
            return
        
        # Determine eviction targets
        if target_size is None:
            target_size = max(0, self.max_size - 1)
        if target_memory is None:
            target_memory = int(self.max_memory_bytes * 0.8)  # Target 80% of max memory
        
        # Sort items by eviction priority (LRU with access frequency consideration)
        items_by_priority = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Calculate priority score (lower = higher priority for eviction)
            time_since_access = current_time - entry.last_access_time
            access_frequency = entry.access_count / max(1, current_time - entry.creation_time)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Priority score: higher time since access, lower frequency, larger size = higher eviction priority
            priority = time_since_access - (access_frequency * 100) + size_penalty
            items_by_priority.append((priority, key, entry))
        
        # Sort by priority (highest priority for eviction first)
        items_by_priority.sort(reverse=True)
        
        # Evict items until we meet targets
        for priority, key, entry in items_by_priority:
            if (len(self._cache) <= target_size and 
                self._stats.memory_usage_bytes <= target_memory):
                break
            
            # Remove item
            del self._cache[key]
            self._stats.memory_usage_bytes -= entry.size_bytes
            self._stats.evictions += 1
            self._stats.cache_size = len(self._cache)
    
    def _cleanup_if_needed(self):
        """Perform cleanup if memory pressure is high."""
        with self._lock:
            if self._should_evict_for_memory() or len(self._cache) > self.max_size:
                self._evict_items()
                
                # Force garbage collection if memory pressure is very high
                memory_pressure = self._get_memory_pressure()
                if memory_pressure > 0.9:
                    gc.collect()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                self._update_hit_rate()
                return None
            
            # Update access statistics
            entry.update_access()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._update_hit_rate()
            
            return entry.data
    
    def put(self, key: Any, data: Any) -> bool:
        """Put item in cache. Returns True if successfully cached."""
        with self._lock:
            # Estimate size
            size_bytes = self._estimate_size(data)
            
            # Check if item is too large for cache
            if size_bytes > self.max_memory_bytes:
                warnings.warn(
                    f"Item size ({size_bytes / (1024*1024):.1f}MB) exceeds cache limit "
                    f"({self.max_memory_bytes / (1024*1024):.1f}MB). Not caching.",
                    UserWarning,
                    stacklevel=2,
                )
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.memory_usage_bytes -= old_entry.size_bytes
                del self._cache[key]
            
            # Create new entry
            entry = EpisodeCacheEntry(
                episode_id=key,
                data=data,
                size_bytes=size_bytes,
                access_count=1,
                last_access_time=time.time(),
                creation_time=time.time(),
            )
            
            # Check if we need to evict items
            projected_memory = self._stats.memory_usage_bytes + size_bytes
            if (len(self._cache) >= self.max_size or 
                projected_memory > self.max_memory_bytes):
                self._evict_items()
            
            # Add new entry
            self._cache[key] = entry
            self._stats.memory_usage_bytes += size_bytes
            self._stats.cache_size = len(self._cache)
            
            return True
    
    def _update_hit_rate(self):
        """Update hit rate statistics."""
        total_requests = self._stats.hits + self._stats.misses
        if total_requests > 0:
            self._stats.hit_rate = self._stats.hits / total_requests
        else:
            self._stats.hit_rate = 0.0
        
        if self.max_memory_bytes > 0:
            self._stats.memory_utilization = self._stats.memory_usage_bytes / self.max_memory_bytes
    
    def clear(self):
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._stats.memory_usage_bytes = 0
            self._stats.cache_size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._update_hit_rate()
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                memory_usage_bytes=self._stats.memory_usage_bytes,
                max_memory_bytes=self._stats.max_memory_bytes,
                cache_size=self._stats.cache_size,
                max_cache_size=self._stats.max_cache_size,
                hit_rate=self._stats.hit_rate,
                memory_utilization=self._stats.memory_utilization,
            )
    
    def reset_stats(self):
        """Reset cache statistics."""
        with self._lock:
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0
            self._update_hit_rate()
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        if self._cleanup_thread is not None:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
        
        self.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


class DistributedEpisodeCache:
    """
    Distributed episode cache that coordinates caching across multiple worker processes.
    
    This cache provides:
    - Per-process local caching with memory awareness
    - Cross-process cache coordination
    - Load balancing and cache warming
    """
    
    def __init__(
        self,
        max_size_per_worker: int = 64,
        max_memory_mb_per_worker: int = 512,
        enable_cross_process_sharing: bool = True,
        cache_warming_enabled: bool = True,
    ):
        """
        Initialize distributed episode cache.
        
        Args:
            max_size_per_worker: Maximum cache size per worker process
            max_memory_mb_per_worker: Maximum memory per worker in MB
            enable_cross_process_sharing: Whether to enable cross-process coordination
            cache_warming_enabled: Whether to enable cache warming
        """
        self.max_size_per_worker = max_size_per_worker
        self.max_memory_mb_per_worker = max_memory_mb_per_worker
        self.enable_cross_process_sharing = enable_cross_process_sharing
        self.cache_warming_enabled = cache_warming_enabled
        
        # Per-process cache instances
        self._local_caches: Dict[int, MemoryAwareLRUCache] = {}
        self._cache_lock = threading.RLock()
        
        # Cross-process coordination (if enabled)
        self._shared_stats = None
        if enable_cross_process_sharing:
            try:
                # Use multiprocessing Manager for cross-process coordination
                self._manager = mp.Manager()
                self._shared_stats = self._manager.dict()
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize cross-process sharing: {e}. "
                    f"Using local-only caching.",
                    UserWarning,
                    stacklevel=2,
                )
                self.enable_cross_process_sharing = False
    
    def _get_local_cache(self) -> MemoryAwareLRUCache:
        """Get or create local cache for current process."""
        process_id = os.getpid()
        
        with self._cache_lock:
            if process_id not in self._local_caches:
                self._local_caches[process_id] = MemoryAwareLRUCache(
                    max_size=self.max_size_per_worker,
                    max_memory_mb=self.max_memory_mb_per_worker,
                    enable_memory_monitoring=True,
                )
            
            return self._local_caches[process_id]
    
    def get(self, episode_id: int, loader_func: callable) -> Any:
        """
        Get episode data from cache or load using loader function.
        
        Args:
            episode_id: Episode identifier
            loader_func: Function to load episode if not in cache
            
        Returns:
            Episode data
        """
        local_cache = self._get_local_cache()
        
        # Try local cache first
        data = local_cache.get(episode_id)
        if data is not None:
            return data
        
        # Load data using loader function
        try:
            data = loader_func(episode_id)
            
            # Cache the loaded data
            success = local_cache.put(episode_id, data)
            
            # Update cross-process statistics
            if self.enable_cross_process_sharing and self._shared_stats is not None:
                try:
                    process_id = os.getpid()
                    self._shared_stats[f"loaded_{process_id}"] = (
                        self._shared_stats.get(f"loaded_{process_id}", 0) + 1
                    )
                    if success:
                        self._shared_stats[f"cached_{process_id}"] = (
                            self._shared_stats.get(f"cached_{process_id}", 0) + 1
                        )
                except Exception:
                    pass  # Ignore cross-process coordination errors
            
            return data
            
        except Exception as e:
            warnings.warn(
                f"Failed to load episode {episode_id}: {e}",
                UserWarning,
                stacklevel=2,
            )
            raise
    
    def prefetch(self, episode_ids: List[int], loader_func: callable):
        """
        Prefetch episodes into cache (non-blocking).
        
        Args:
            episode_ids: List of episode IDs to prefetch
            loader_func: Function to load episodes
        """
        if not self.cache_warming_enabled:
            return
        
        local_cache = self._get_local_cache()
        
        # Only prefetch episodes not already in cache
        to_prefetch = [
            episode_id for episode_id in episode_ids
            if local_cache.get(episode_id) is None
        ]
        
        if not to_prefetch:
            return
        
        # Prefetch in background thread to avoid blocking
        def _prefetch_worker():
            for episode_id in to_prefetch:
                try:
                    if local_cache.get(episode_id) is None:  # Double-check
                        data = loader_func(episode_id)
                        local_cache.put(episode_id, data)
                except Exception as e:
                    # Ignore prefetch errors to avoid disrupting main flow
                    pass
        
        thread = threading.Thread(target=_prefetch_worker, daemon=True)
        thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {}
        
        # Local cache stats
        with self._cache_lock:
            for process_id, cache in self._local_caches.items():
                cache_stats = cache.get_stats()
                stats[f"process_{process_id}"] = {
                    "hits": cache_stats.hits,
                    "misses": cache_stats.misses,
                    "evictions": cache_stats.evictions,
                    "hit_rate": cache_stats.hit_rate,
                    "memory_usage_mb": cache_stats.memory_usage_bytes / (1024 * 1024),
                    "memory_utilization": cache_stats.memory_utilization,
                    "cache_size": cache_stats.cache_size,
                }
        
        # Cross-process stats
        if self.enable_cross_process_sharing and self._shared_stats is not None:
            try:
                stats["cross_process"] = dict(self._shared_stats)
            except Exception:
                pass
        
        # Aggregate stats
        total_hits = sum(s.get("hits", 0) for s in stats.values() if isinstance(s, dict))
        total_misses = sum(s.get("misses", 0) for s in stats.values() if isinstance(s, dict))
        total_requests = total_hits + total_misses
        
        stats["aggregate"] = {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_requests": total_requests,
            "overall_hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "active_processes": len(self._local_caches),
        }
        
        return stats
    
    def reset_stats(self):
        """Reset all cache statistics."""
        with self._cache_lock:
            for cache in self._local_caches.values():
                cache.reset_stats()
        
        if self.enable_cross_process_sharing and self._shared_stats is not None:
            try:
                self._shared_stats.clear()
            except Exception:
                pass
    
    def clear_all(self):
        """Clear all caches."""
        with self._cache_lock:
            for cache in self._local_caches.values():
                cache.clear()
    
    def shutdown(self):
        """Shutdown all caches and cleanup resources."""
        with self._cache_lock:
            for cache in self._local_caches.values():
                cache.shutdown()
            self._local_caches.clear()
        
        if hasattr(self, '_manager'):
            try:
                self._manager.shutdown()
            except Exception:
                pass


# Global distributed cache instance
_global_distributed_cache: Optional[DistributedEpisodeCache] = None


def get_global_episode_cache(
    max_size_per_worker: int = 64,
    max_memory_mb_per_worker: int = 512,
    **kwargs
) -> DistributedEpisodeCache:
    """
    Get or create global distributed episode cache.
    
    Args:
        max_size_per_worker: Maximum cache size per worker process
        max_memory_mb_per_worker: Maximum memory per worker in MB
        **kwargs: Additional arguments for DistributedEpisodeCache
        
    Returns:
        Global distributed episode cache instance
    """
    global _global_distributed_cache
    
    if _global_distributed_cache is None:
        _global_distributed_cache = DistributedEpisodeCache(
            max_size_per_worker=max_size_per_worker,
            max_memory_mb_per_worker=max_memory_mb_per_worker,
            **kwargs
        )
    
    return _global_distributed_cache


def shutdown_global_episode_cache():
    """Shutdown global episode cache."""
    global _global_distributed_cache
    
    if _global_distributed_cache is not None:
        _global_distributed_cache.shutdown()
        _global_distributed_cache = None


def create_cached_loader(
    loader_func: callable,
    cache_config: Optional[Dict[str, Any]] = None,
) -> callable:
    """
    Create a cached version of an episode loader function.
    
    Args:
        loader_func: Original loader function that takes episode_id and returns data
        cache_config: Optional cache configuration
        
    Returns:
        Cached loader function
    """
    cache_config = cache_config or {}
    cache = get_global_episode_cache(**cache_config)
    
    def cached_loader(episode_id: int):
        return cache.get(episode_id, loader_func)
    
    # Add cache management methods
    cached_loader.get_cache_stats = cache.get_stats
    cached_loader.reset_cache_stats = cache.reset_stats
    cached_loader.clear_cache = cache.clear_all
    cached_loader.prefetch = lambda episode_ids: cache.prefetch(episode_ids, loader_func)
    
    return cached_loader