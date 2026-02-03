"""
Tests for intelligent episode cache.
"""

import time
import threading
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import pytest

from openpi.training.intelligent_episode_cache import (
    MemoryAwareLRUCache,
    DistributedEpisodeCache,
    EpisodeCacheEntry,
    CacheStats,
    get_global_episode_cache,
    shutdown_global_episode_cache,
    create_cached_loader,
)


class TestEpisodeCacheEntry:
    """Test cases for EpisodeCacheEntry."""
    
    def test_creation(self):
        """Test entry creation."""
        data = {"test": "data"}
        entry = EpisodeCacheEntry(
            episode_id=1,
            data=data,
            size_bytes=1024,
            access_count=1,
            last_access_time=time.time(),
            creation_time=time.time(),
        )
        
        assert entry.episode_id == 1
        assert entry.data == data
        assert entry.size_bytes == 1024
        assert entry.access_count == 1
    
    def test_update_access(self):
        """Test access update."""
        entry = EpisodeCacheEntry(
            episode_id=1,
            data={},
            size_bytes=1024,
            access_count=1,
            last_access_time=time.time() - 10,
            creation_time=time.time() - 10,
        )
        
        old_access_time = entry.last_access_time
        old_access_count = entry.access_count
        
        entry.update_access()
        
        assert entry.access_count == old_access_count + 1
        assert entry.last_access_time > old_access_time


class TestMemoryAwareLRUCache:
    """Test cases for MemoryAwareLRUCache."""
    
    def test_init(self):
        """Test cache initialization."""
        cache = MemoryAwareLRUCache(
            max_size=10,
            max_memory_mb=100,
            memory_pressure_threshold=0.8,
            enable_memory_monitoring=False,  # Disable for testing
        )
        
        assert cache.max_size == 10
        assert cache.max_memory_bytes == 100 * 1024 * 1024
        assert cache.memory_pressure_threshold == 0.8
        assert cache.enable_memory_monitoring is False
    
    def test_estimate_size_dataframe(self):
        """Test size estimation for DataFrame."""
        cache = MemoryAwareLRUCache(enable_memory_monitoring=False)
        
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        size = cache._estimate_size(df)
        assert size > 0
        assert isinstance(size, int)
    
    def test_estimate_size_numpy(self):
        """Test size estimation for numpy array."""
        cache = MemoryAwareLRUCache(enable_memory_monitoring=False)
        
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        size = cache._estimate_size(arr)
        
        assert size == arr.nbytes
    
    def test_estimate_size_dict(self):
        """Test size estimation for dictionary."""
        cache = MemoryAwareLRUCache(enable_memory_monitoring=False)
        
        data = {
            'array': np.array([1, 2, 3]),
            'string': 'test',
            'number': 42
        }
        
        size = cache._estimate_size(data)
        assert size > 0
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = MemoryAwareLRUCache(
            max_size=5,
            max_memory_mb=10,
            enable_memory_monitoring=False,
        )
        
        data = "test_data"
        success = cache.put("key1", data)
        
        assert success is True
        assert cache.get("key1") == data
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when max size is exceeded."""
        cache = MemoryAwareLRUCache(
            max_size=2,
            max_memory_mb=100,
            enable_memory_monitoring=False,
        )
        
        # Add items up to capacity
        cache.put("key1", "data1")
        cache.put("key2", "data2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add another item, should evict key2 (least recently used)
        cache.put("key3", "data3")
        
        assert cache.get("key1") == "data1"  # Should still be there
        assert cache.get("key2") is None     # Should be evicted
        assert cache.get("key3") == "data3"  # Should be there
    
    def test_memory_limit_eviction(self):
        """Test eviction when memory limit is exceeded."""
        cache = MemoryAwareLRUCache(
            max_size=10,
            max_memory_mb=1,  # Very small memory limit
            enable_memory_monitoring=False,
        )
        
        # Create large data that exceeds memory limit
        large_data = "x" * (512 * 1024)  # 512KB string
        
        cache.put("key1", large_data)
        cache.put("key2", large_data)
        cache.put("key3", large_data)  # Should trigger eviction
        
        # Some items should be evicted due to memory pressure
        cached_items = sum(1 for key in ["key1", "key2", "key3"] if cache.get(key) is not None)
        assert cached_items < 3
    
    def test_item_too_large(self):
        """Test handling of items too large for cache."""
        cache = MemoryAwareLRUCache(
            max_size=10,
            max_memory_mb=1,
            enable_memory_monitoring=False,
        )
        
        # Create data larger than cache limit
        huge_data = "x" * (2 * 1024 * 1024)  # 2MB string
        
        success = cache.put("huge", huge_data)
        assert success is False
        assert cache.get("huge") is None
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = MemoryAwareLRUCache(enable_memory_monitoring=False)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        
        # Add item and access it
        cache.put("key1", "data1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5
    
    def test_clear(self):
        """Test cache clearing."""
        cache = MemoryAwareLRUCache(enable_memory_monitoring=False)
        
        cache.put("key1", "data1")
        cache.put("key2", "data2")
        
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        
        stats = cache.get_stats()
        assert stats.cache_size == 0
        assert stats.memory_usage_bytes == 0
    
    def test_reset_stats(self):
        """Test statistics reset."""
        cache = MemoryAwareLRUCache(enable_memory_monitoring=False)
        
        cache.put("key1", "data1")
        cache.get("key1")
        cache.get("key2")
        
        stats = cache.get_stats()
        assert stats.hits > 0
        assert stats.misses > 0
        
        cache.reset_stats()
        
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0


class TestDistributedEpisodeCache:
    """Test cases for DistributedEpisodeCache."""
    
    def test_init(self):
        """Test cache initialization."""
        cache = DistributedEpisodeCache(
            max_size_per_worker=32,
            max_memory_mb_per_worker=256,
            enable_cross_process_sharing=False,  # Disable for testing
            cache_warming_enabled=True,
        )
        
        assert cache.max_size_per_worker == 32
        assert cache.max_memory_mb_per_worker == 256
        assert cache.enable_cross_process_sharing is False
        assert cache.cache_warming_enabled is True
    
    def test_get_with_loader(self):
        """Test get operation with loader function."""
        cache = DistributedEpisodeCache(
            enable_cross_process_sharing=False,
            cache_warming_enabled=False,
        )
        
        # Mock loader function
        loader_calls = []
        def mock_loader(episode_id):
            loader_calls.append(episode_id)
            return f"data_for_episode_{episode_id}"
        
        # First call should invoke loader
        result1 = cache.get(1, mock_loader)
        assert result1 == "data_for_episode_1"
        assert loader_calls == [1]
        
        # Second call should use cache
        result2 = cache.get(1, mock_loader)
        assert result2 == "data_for_episode_1"
        assert loader_calls == [1]  # No additional calls
    
    def test_get_different_episodes(self):
        """Test get operation with different episodes."""
        cache = DistributedEpisodeCache(
            enable_cross_process_sharing=False,
            cache_warming_enabled=False,
        )
        
        def mock_loader(episode_id):
            return f"data_{episode_id}"
        
        result1 = cache.get(1, mock_loader)
        result2 = cache.get(2, mock_loader)
        
        assert result1 == "data_1"
        assert result2 == "data_2"
        
        # Both should be cached
        assert cache.get(1, mock_loader) == "data_1"
        assert cache.get(2, mock_loader) == "data_2"
    
    def test_prefetch(self):
        """Test prefetch functionality."""
        cache = DistributedEpisodeCache(
            enable_cross_process_sharing=False,
            cache_warming_enabled=True,
        )
        
        loader_calls = []
        def mock_loader(episode_id):
            loader_calls.append(episode_id)
            return f"data_{episode_id}"
        
        # Prefetch episodes
        cache.prefetch([1, 2, 3], mock_loader)
        
        # Give some time for background prefetch
        time.sleep(0.1)
        
        # Episodes should be cached (no additional loader calls)
        result1 = cache.get(1, mock_loader)
        result2 = cache.get(2, mock_loader)
        
        assert result1 == "data_1"
        assert result2 == "data_2"
        
        # Check that prefetch actually loaded the data
        assert 1 in loader_calls
        assert 2 in loader_calls
    
    def test_prefetch_disabled(self):
        """Test prefetch when cache warming is disabled."""
        cache = DistributedEpisodeCache(
            enable_cross_process_sharing=False,
            cache_warming_enabled=False,
        )
        
        loader_calls = []
        def mock_loader(episode_id):
            loader_calls.append(episode_id)
            return f"data_{episode_id}"
        
        # Prefetch should do nothing
        cache.prefetch([1, 2, 3], mock_loader)
        time.sleep(0.1)
        
        assert loader_calls == []
        
        # Episodes should not be cached
        cache.get(1, mock_loader)
        assert loader_calls == [1]
    
    def test_get_stats(self):
        """Test statistics collection."""
        cache = DistributedEpisodeCache(
            enable_cross_process_sharing=False,
            cache_warming_enabled=False,
        )
        
        def mock_loader(episode_id):
            return f"data_{episode_id}"
        
        # Generate some cache activity
        cache.get(1, mock_loader)
        cache.get(1, mock_loader)  # Cache hit
        cache.get(2, mock_loader)
        
        stats = cache.get_stats()
        
        assert "aggregate" in stats
        assert stats["aggregate"]["total_requests"] > 0
        assert stats["aggregate"]["overall_hit_rate"] >= 0.0
    
    def test_clear_all(self):
        """Test clearing all caches."""
        cache = DistributedEpisodeCache(
            enable_cross_process_sharing=False,
            cache_warming_enabled=False,
        )
        
        def mock_loader(episode_id):
            return f"data_{episode_id}"
        
        # Add some data
        cache.get(1, mock_loader)
        cache.get(2, mock_loader)
        
        # Clear all caches
        cache.clear_all()
        
        # Data should be reloaded
        loader_calls = []
        def counting_loader(episode_id):
            loader_calls.append(episode_id)
            return f"data_{episode_id}"
        
        cache.get(1, counting_loader)
        assert loader_calls == [1]


class TestGlobalCache:
    """Test cases for global cache functions."""
    
    def test_get_global_cache(self):
        """Test global cache creation."""
        # Clean up any existing cache
        shutdown_global_episode_cache()
        
        cache1 = get_global_episode_cache(max_size_per_worker=32)
        cache2 = get_global_episode_cache(max_size_per_worker=64)
        
        # Should return same instance
        assert cache1 is cache2
        assert cache1.max_size_per_worker == 32  # Original settings preserved
    
    def test_shutdown_global_cache(self):
        """Test global cache shutdown."""
        cache = get_global_episode_cache()
        cache.shutdown = Mock()
        
        shutdown_global_episode_cache()
        
        cache.shutdown.assert_called_once()
        
        # Next call should create new cache
        new_cache = get_global_episode_cache()
        assert new_cache is not cache


class TestCachedLoader:
    """Test cases for cached loader creation."""
    
    def test_create_cached_loader(self):
        """Test cached loader creation."""
        # Clean up global cache
        shutdown_global_episode_cache()
        
        original_calls = []
        def original_loader(episode_id):
            original_calls.append(episode_id)
            return f"data_{episode_id}"
        
        cached_loader = create_cached_loader(original_loader)
        
        # First call should invoke original loader
        result1 = cached_loader(1)
        assert result1 == "data_1"
        assert original_calls == [1]
        
        # Second call should use cache
        result2 = cached_loader(1)
        assert result2 == "data_1"
        assert original_calls == [1]  # No additional calls
    
    def test_cached_loader_methods(self):
        """Test cached loader additional methods."""
        def original_loader(episode_id):
            return f"data_{episode_id}"
        
        cached_loader = create_cached_loader(original_loader)
        
        # Test that additional methods are available
        assert hasattr(cached_loader, 'get_cache_stats')
        assert hasattr(cached_loader, 'reset_cache_stats')
        assert hasattr(cached_loader, 'clear_cache')
        assert hasattr(cached_loader, 'prefetch')
        
        # Test stats method
        stats = cached_loader.get_cache_stats()
        assert isinstance(stats, dict)
        
        # Test prefetch method
        cached_loader.prefetch([1, 2, 3])  # Should not raise error


class TestCacheStats:
    """Test cases for CacheStats dataclass."""
    
    def test_creation(self):
        """Test stats creation."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            memory_usage_bytes=1024,
            max_memory_bytes=2048,
            cache_size=8,
            max_cache_size=10,
            hit_rate=0.67,
            memory_utilization=0.5,
        )
        
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.memory_usage_bytes == 1024
        assert stats.max_memory_bytes == 2048
        assert stats.cache_size == 8
        assert stats.max_cache_size == 10
        assert stats.hit_rate == 0.67
        assert stats.memory_utilization == 0.5