"""
Tests for video backend optimizer.
"""

import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from openpi.training.video_backend_optimizer import (
    VideoBackendConfig,
    VideoBackendOptimizer,
    VideoDecodeStats,
    get_global_video_optimizer,
    shutdown_global_video_optimizer,
    create_optimized_video_backend_kwargs,
)


class TestVideoBackendConfig:
    """Test cases for VideoBackendConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = VideoBackendConfig()
        
        assert config.backend == "torchcodec"
        assert config.num_threads is None
        assert config.thread_type == "FRAME"
        assert config.pixel_format == "rgb24"
        assert config.hardware_acceleration is True
        assert config.enable_frame_cache is True
        assert config.frame_cache_size_mb == 256
        assert config.prefetch_frames == 8
        assert config.enable_performance_monitoring is True
        assert config.log_slow_decodes is True
        assert config.slow_decode_threshold_ms == 100.0
        assert config.extra_kwargs == {}
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = VideoBackendConfig(
            backend="opencv",
            num_threads=4,
            hardware_acceleration=False,
            frame_cache_size_mb=512,
            extra_kwargs={"test": "value"},
        )
        
        assert config.backend == "opencv"
        assert config.num_threads == 4
        assert config.hardware_acceleration is False
        assert config.frame_cache_size_mb == 512
        assert config.extra_kwargs == {"test": "value"}


class TestVideoDecodeStats:
    """Test cases for VideoDecodeStats."""
    
    def test_default_stats(self):
        """Test default statistics."""
        stats = VideoDecodeStats()
        
        assert stats.total_decodes == 0
        assert stats.total_decode_time == 0.0
        assert stats.total_frames_decoded == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.slow_decodes == 0
        assert stats.average_decode_time_ms == 0.0
        assert stats.average_frames_per_second == 0.0
        assert stats.cache_hit_rate == 0.0
    
    def test_custom_stats(self):
        """Test custom statistics."""
        stats = VideoDecodeStats(
            total_decodes=10,
            total_decode_time=5.0,
            total_frames_decoded=100,
            cache_hits=8,
            cache_misses=2,
            slow_decodes=1,
        )
        
        assert stats.total_decodes == 10
        assert stats.total_decode_time == 5.0
        assert stats.total_frames_decoded == 100
        assert stats.cache_hits == 8
        assert stats.cache_misses == 2
        assert stats.slow_decodes == 1


class TestVideoBackendOptimizer:
    """Test cases for VideoBackendOptimizer."""
    
    def test_init_default(self):
        """Test optimizer initialization with default config."""
        optimizer = VideoBackendOptimizer()
        
        assert optimizer.config.backend == "torchcodec"
        assert optimizer.config.num_threads is not None  # Should be auto-detected
        assert optimizer.config.num_threads >= 1
    
    def test_init_custom_config(self):
        """Test optimizer initialization with custom config."""
        config = VideoBackendConfig(
            backend="opencv",
            num_threads=4,
            hardware_acceleration=False,
        )
        
        optimizer = VideoBackendOptimizer(config)
        
        assert optimizer.config.backend == "opencv"
        assert optimizer.config.num_threads == 4
        assert optimizer.config.hardware_acceleration is False
    
    @patch('torch.cuda.is_available')
    def test_detect_hardware_acceleration_cuda(self, mock_cuda_available):
        """Test hardware acceleration detection with CUDA."""
        mock_cuda_available.return_value = True
        
        optimizer = VideoBackendOptimizer()
        result = optimizer._detect_hardware_acceleration()
        
        assert result is True
    
    @patch('torch.cuda.is_available')
    def test_detect_hardware_acceleration_no_cuda(self, mock_cuda_available):
        """Test hardware acceleration detection without CUDA."""
        mock_cuda_available.return_value = False
        
        optimizer = VideoBackendOptimizer()
        result = optimizer._detect_hardware_acceleration()
        
        assert result is False
    
    def test_estimate_frame_size_numpy(self):
        """Test frame size estimation for numpy arrays."""
        optimizer = VideoBackendOptimizer()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        size = optimizer._estimate_frame_size(frame)
        
        assert size == frame.nbytes
    
    def test_estimate_frame_size_unknown(self):
        """Test frame size estimation for unknown types."""
        optimizer = VideoBackendOptimizer()
        
        unknown_data = "test"
        size = optimizer._estimate_frame_size(unknown_data)
        
        assert size == 1024 * 1024  # Default 1MB
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        optimizer = VideoBackendOptimizer()
        
        video_path = "/path/to/video.mp4"
        frame_indices = np.array([0, 1, 2])
        
        key = optimizer._cache_key(video_path, frame_indices)
        
        assert key == "/path/to/video.mp4:0,1,2"
    
    def test_cache_frames_disabled(self):
        """Test frame caching when disabled."""
        config = VideoBackendConfig(enable_frame_cache=False)
        optimizer = VideoBackendOptimizer(config)
        
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        optimizer._cache_frames("test_key", frames)
        
        # Should not cache anything
        assert len(optimizer._frame_cache) == 0
    
    def test_cache_frames_enabled(self):
        """Test frame caching when enabled."""
        config = VideoBackendConfig(enable_frame_cache=True, frame_cache_size_mb=10)
        optimizer = VideoBackendOptimizer(config)
        
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        optimizer._cache_frames("test_key", frames)
        
        # Should cache the frames
        assert len(optimizer._frame_cache) == 1
        assert "test_key" in optimizer._frame_cache
    
    def test_get_cached_frames_hit(self):
        """Test getting cached frames (cache hit)."""
        config = VideoBackendConfig(enable_frame_cache=True)
        optimizer = VideoBackendOptimizer(config)
        
        # Cache some frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        optimizer._cache_frames("test_key", frames)
        
        # Retrieve cached frames
        cached_frames = optimizer._get_cached_frames("test_key")
        
        assert cached_frames is not None
        assert len(cached_frames) == 1
        np.testing.assert_array_equal(cached_frames[0], frames[0])
    
    def test_get_cached_frames_miss(self):
        """Test getting cached frames (cache miss)."""
        optimizer = VideoBackendOptimizer()
        
        cached_frames = optimizer._get_cached_frames("nonexistent_key")
        
        assert cached_frames is None
    
    def test_get_cached_frames_disabled(self):
        """Test getting cached frames when caching is disabled."""
        config = VideoBackendConfig(enable_frame_cache=False)
        optimizer = VideoBackendOptimizer(config)
        
        cached_frames = optimizer._get_cached_frames("test_key")
        
        assert cached_frames is None
    
    @patch('openpi.training.video_backend_optimizer.VideoBackendOptimizer._decode_with_backend')
    def test_decode_video_frames_cache_miss(self, mock_decode):
        """Test video frame decoding with cache miss."""
        config = VideoBackendConfig(enable_frame_cache=True)
        optimizer = VideoBackendOptimizer(config)
        
        # Mock decoder
        mock_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        mock_decode.return_value = mock_frames
        
        video_path = "/path/to/video.mp4"
        frame_indices = np.array([0, 1])
        
        result = optimizer.decode_video_frames(video_path, frame_indices)
        
        assert result == mock_frames
        mock_decode.assert_called_once()
        
        # Check statistics
        stats = optimizer.get_stats()
        assert stats.total_decodes == 1
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0
    
    @patch('openpi.training.video_backend_optimizer.VideoBackendOptimizer._decode_with_backend')
    def test_decode_video_frames_cache_hit(self, mock_decode):
        """Test video frame decoding with cache hit."""
        config = VideoBackendConfig(enable_frame_cache=True)
        optimizer = VideoBackendOptimizer(config)
        
        # Pre-cache some frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        video_path = "/path/to/video.mp4"
        frame_indices = np.array([0, 1])
        cache_key = optimizer._cache_key(video_path, frame_indices)
        optimizer._cache_frames(cache_key, frames)
        
        result = optimizer.decode_video_frames(video_path, frame_indices)
        
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], frames[0])
        mock_decode.assert_not_called()  # Should not call decoder
        
        # Check statistics
        stats = optimizer.get_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 0
    
    @patch('openpi.training.video_backend_optimizer.VideoBackendOptimizer._decode_with_backend')
    def test_decode_video_frames_error(self, mock_decode):
        """Test video frame decoding with error."""
        optimizer = VideoBackendOptimizer()
        
        # Mock decoder to raise error
        mock_decode.side_effect = Exception("Decode error")
        
        video_path = "/path/to/video.mp4"
        frame_indices = np.array([0, 1])
        
        with pytest.raises(RuntimeError, match="Video decoding failed"):
            optimizer.decode_video_frames(video_path, frame_indices)
        
        # Check statistics (should still be updated)
        stats = optimizer.get_stats()
        assert stats.total_decodes == 1
    
    def test_decode_with_backend_unsupported(self):
        """Test decoding with unsupported backend."""
        config = VideoBackendConfig(backend="unsupported")
        optimizer = VideoBackendOptimizer(config)
        
        with pytest.raises(ValueError, match="Unsupported video backend"):
            optimizer._decode_with_backend("/path/to/video.mp4", np.array([0]), {})
    
    @patch('threading.Thread')
    def test_prefetch_frames(self, mock_thread):
        """Test frame prefetching."""
        config = VideoBackendConfig(enable_frame_cache=True)
        optimizer = VideoBackendOptimizer(config)
        
        video_path = "/path/to/video.mp4"
        frame_indices = np.array([0, 1])
        
        optimizer.prefetch_frames(video_path, frame_indices)
        
        # Should start a background thread
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
    
    def test_prefetch_frames_disabled(self):
        """Test frame prefetching when caching is disabled."""
        config = VideoBackendConfig(enable_frame_cache=False)
        optimizer = VideoBackendOptimizer(config)
        
        # Should not raise error, but also not do anything
        optimizer.prefetch_frames("/path/to/video.mp4", np.array([0, 1]))
    
    def test_get_stats_empty(self):
        """Test getting statistics with no activity."""
        optimizer = VideoBackendOptimizer()
        
        stats = optimizer.get_stats()
        
        assert stats.total_decodes == 0
        assert stats.total_decode_time == 0.0
        assert stats.total_frames_decoded == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.slow_decodes == 0
        assert stats.average_decode_time_ms == 0.0
        assert stats.average_frames_per_second == 0.0
        assert stats.cache_hit_rate == 0.0
    
    def test_reset_stats(self):
        """Test statistics reset."""
        optimizer = VideoBackendOptimizer()
        
        # Simulate some activity
        optimizer._stats.total_decodes = 5
        optimizer._stats.cache_hits = 3
        optimizer._stats.cache_misses = 2
        
        optimizer.reset_stats()
        
        stats = optimizer.get_stats()
        assert stats.total_decodes == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
    
    def test_clear_cache(self):
        """Test cache clearing."""
        config = VideoBackendConfig(enable_frame_cache=True)
        optimizer = VideoBackendOptimizer(config)
        
        # Add some cached frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        optimizer._cache_frames("test_key", frames)
        
        assert len(optimizer._frame_cache) == 1
        assert optimizer._current_cache_size > 0
        
        optimizer.clear_cache()
        
        assert len(optimizer._frame_cache) == 0
        assert optimizer._current_cache_size == 0
    
    def test_get_cache_info(self):
        """Test cache information retrieval."""
        config = VideoBackendConfig(enable_frame_cache=True, frame_cache_size_mb=10)
        optimizer = VideoBackendOptimizer(config)
        
        cache_info = optimizer.get_cache_info()
        
        assert "cache_size_mb" in cache_info
        assert "max_cache_size_mb" in cache_info
        assert "cached_items" in cache_info
        assert "cache_utilization" in cache_info
        
        assert cache_info["max_cache_size_mb"] == 10
        assert cache_info["cached_items"] == 0
        assert cache_info["cache_utilization"] == 0.0


class TestGlobalOptimizer:
    """Test cases for global optimizer functions."""
    
    def test_get_global_optimizer(self):
        """Test global optimizer creation."""
        # Clean up any existing optimizer
        shutdown_global_video_optimizer()
        
        optimizer1 = get_global_video_optimizer()
        optimizer2 = get_global_video_optimizer()
        
        # Should return same instance
        assert optimizer1 is optimizer2
    
    def test_get_global_optimizer_with_config(self):
        """Test global optimizer creation with config."""
        shutdown_global_video_optimizer()
        
        config = VideoBackendConfig(backend="opencv", num_threads=4)
        optimizer = get_global_video_optimizer(config)
        
        assert optimizer.config.backend == "opencv"
        assert optimizer.config.num_threads == 4
    
    def test_shutdown_global_optimizer(self):
        """Test global optimizer shutdown."""
        optimizer = get_global_video_optimizer()
        optimizer.clear_cache = Mock()
        
        shutdown_global_video_optimizer()
        
        optimizer.clear_cache.assert_called_once()
        
        # Next call should create new optimizer
        new_optimizer = get_global_video_optimizer()
        assert new_optimizer is not optimizer


class TestOptimizedKwargs:
    """Test cases for optimized kwargs creation."""
    
    def test_create_optimized_kwargs_default(self):
        """Test creating optimized kwargs with defaults."""
        kwargs = create_optimized_video_backend_kwargs()
        
        assert isinstance(kwargs, dict)
        assert "num_threads" in kwargs
        assert kwargs["num_threads"] >= 1
    
    def test_create_optimized_kwargs_custom(self):
        """Test creating optimized kwargs with custom settings."""
        kwargs = create_optimized_video_backend_kwargs(
            backend="opencv",
            enable_hardware_acceleration=False,
            custom_param="value",
        )
        
        assert isinstance(kwargs, dict)
        assert "custom_param" in kwargs
        assert kwargs["custom_param"] == "value"
    
    @patch('torch.cuda.is_available')
    def test_create_optimized_kwargs_cuda(self, mock_cuda_available):
        """Test creating optimized kwargs with CUDA."""
        mock_cuda_available.return_value = True
        
        kwargs = create_optimized_video_backend_kwargs(
            backend="torchcodec",
            enable_hardware_acceleration=True,
        )
        
        assert "device" in kwargs
        assert kwargs["device"] == "cuda"