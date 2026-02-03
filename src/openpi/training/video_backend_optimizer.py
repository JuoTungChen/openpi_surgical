"""
Video backend optimization for GR00T dataset processing.

This module provides optimized video decoding configurations and performance
monitoring to reduce video processing bottlenecks in the data pipeline.
"""

from __future__ import annotations

import os
import time
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import multiprocessing as mp

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError("openpi requires torch for data loading.") from e


@dataclass
class VideoBackendConfig:
    """Configuration for optimized video backend."""
    backend: str = "torchcodec"
    # TorchCodec specific settings
    num_threads: Optional[int] = None  # Auto-detect based on CPU count
    thread_type: str = "FRAME"  # or "SLICE"
    # Decoding optimization
    pixel_format: str = "rgb24"
    hardware_acceleration: bool = True
    # Caching and prefetching
    enable_frame_cache: bool = True
    frame_cache_size_mb: int = 256
    prefetch_frames: int = 8
    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_slow_decodes: bool = True
    slow_decode_threshold_ms: float = 100.0
    # Additional backend kwargs
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoDecodeStats:
    """Statistics for video decoding performance."""
    total_decodes: int = 0
    total_decode_time: float = 0.0
    total_frames_decoded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    slow_decodes: int = 0
    average_decode_time_ms: float = 0.0
    average_frames_per_second: float = 0.0
    cache_hit_rate: float = 0.0


class VideoBackendOptimizer:
    """
    Optimizer for video backend configuration and performance monitoring.
    
    This class provides:
    - Automatic detection of optimal video backend settings
    - Performance monitoring and statistics
    - Frame caching and prefetching
    - Hardware acceleration detection
    """
    
    def __init__(self, config: Optional[VideoBackendConfig] = None):
        """
        Initialize video backend optimizer.
        
        Args:
            config: Video backend configuration (None for auto-detection)
        """
        self.config = config or VideoBackendConfig()
        self._stats = VideoDecodeStats()
        self._stats_lock = threading.Lock()
        
        # Frame cache
        self._frame_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = self.config.frame_cache_size_mb * 1024 * 1024
        self._current_cache_size = 0
        
        # Auto-configure if needed
        self._auto_configure()
    
    def _auto_configure(self):
        """Auto-configure video backend settings based on system capabilities."""
        # Auto-detect number of threads
        if self.config.num_threads is None:
            cpu_count = mp.cpu_count() or 1
            # Use 75% of available cores for video decoding
            self.config.num_threads = max(1, int(cpu_count * 0.75))
        
        # Detect hardware acceleration capabilities
        if self.config.hardware_acceleration:
            self.config.hardware_acceleration = self._detect_hardware_acceleration()
        
        # Optimize backend kwargs
        self._optimize_backend_kwargs()
    
    def _detect_hardware_acceleration(self) -> bool:
        """Detect if hardware acceleration is available."""
        try:
            # Check for NVIDIA GPU
            if torch.cuda.is_available():
                return True
            
            # Check for other hardware acceleration (Intel QSV, etc.)
            # This is a simplified check - in practice, you'd want more thorough detection
            return False
            
        except Exception:
            return False
    
    def _optimize_backend_kwargs(self):
        """Optimize backend-specific kwargs based on configuration."""
        if self.config.backend == "torchcodec":
            # TorchCodec optimizations
            kwargs = {
                "num_threads": self.config.num_threads,
                "thread_type": self.config.thread_type,
            }
            
            # Add hardware acceleration if available
            if self.config.hardware_acceleration:
                kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Merge with extra kwargs
            kwargs.update(self.config.extra_kwargs)
            self.config.extra_kwargs = kwargs
            
        elif self.config.backend == "opencv":
            # OpenCV optimizations
            kwargs = {
                "backend": "CAP_FFMPEG",
                "num_threads": self.config.num_threads,
            }
            kwargs.update(self.config.extra_kwargs)
            self.config.extra_kwargs = kwargs
            
        elif self.config.backend == "decord":
            # Decord optimizations
            kwargs = {
                "num_threads": self.config.num_threads,
                "ctx": "gpu" if self.config.hardware_acceleration and torch.cuda.is_available() else "cpu",
            }
            kwargs.update(self.config.extra_kwargs)
            self.config.extra_kwargs = kwargs
    
    def _estimate_frame_size(self, frame: np.ndarray) -> int:
        """Estimate memory size of a video frame."""
        if isinstance(frame, np.ndarray):
            return frame.nbytes
        else:
            # Conservative estimate for unknown types
            return 1024 * 1024  # 1MB
    
    def _cache_key(self, video_path: str, frame_indices: np.ndarray) -> str:
        """Generate cache key for video frames."""
        indices_str = ",".join(map(str, frame_indices))
        return f"{video_path}:{indices_str}"
    
    def _evict_cache_items(self, target_size: int):
        """Evict items from frame cache to reach target size."""
        if not self._frame_cache:
            return
        
        # Sort by access time (LRU eviction)
        items_by_time = [
            (access_time, key, frame)
            for key, (frame, access_time) in self._frame_cache.items()
        ]
        items_by_time.sort()  # Oldest first
        
        # Evict oldest items until we reach target size
        for access_time, key, frame in items_by_time:
            if self._current_cache_size <= target_size:
                break
            
            frame_size = self._estimate_frame_size(frame)
            del self._frame_cache[key]
            self._current_cache_size -= frame_size
    
    def _cache_frames(self, cache_key: str, frames: List[np.ndarray]):
        """Cache decoded frames."""
        if not self.config.enable_frame_cache or not frames:
            return
        
        with self._cache_lock:
            # Estimate total size of frames
            total_size = sum(self._estimate_frame_size(frame) for frame in frames)
            
            # Check if frames fit in cache
            if total_size > self._max_cache_size:
                return  # Too large to cache
            
            # Evict items if necessary
            if self._current_cache_size + total_size > self._max_cache_size:
                target_size = self._max_cache_size - total_size
                self._evict_cache_items(target_size)
            
            # Cache frames (store as single array for efficiency)
            if len(frames) == 1:
                cached_data = frames[0]
            else:
                cached_data = np.stack(frames, axis=0)
            
            self._frame_cache[cache_key] = (cached_data, time.time())
            self._current_cache_size += total_size
    
    def _get_cached_frames(self, cache_key: str) -> Optional[List[np.ndarray]]:
        """Get frames from cache."""
        if not self.config.enable_frame_cache:
            return None
        
        with self._cache_lock:
            cached_item = self._frame_cache.get(cache_key)
            if cached_item is None:
                return None
            
            cached_data, _ = cached_item
            # Update access time
            self._frame_cache[cache_key] = (cached_data, time.time())
            
            # Convert back to list of frames
            if cached_data.ndim == 3:  # Single frame
                return [cached_data]
            else:  # Multiple frames
                return [cached_data[i] for i in range(cached_data.shape[0])]
    
    def decode_video_frames(
        self,
        video_path: str,
        frame_indices: np.ndarray,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Decode video frames with optimization and caching.
        
        Args:
            video_path: Path to video file
            frame_indices: Array of frame indices to decode
            **kwargs: Additional arguments for video decoder
            
        Returns:
            List of decoded frames as numpy arrays
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._cache_key(video_path, frame_indices)
        
        # Try cache first
        cached_frames = self._get_cached_frames(cache_key)
        if cached_frames is not None:
            with self._stats_lock:
                self._stats.cache_hits += 1
            return cached_frames
        
        # Cache miss - decode frames
        with self._stats_lock:
            self._stats.cache_misses += 1
        
        try:
            # Merge config kwargs with provided kwargs
            decode_kwargs = self.config.extra_kwargs.copy()
            decode_kwargs.update(kwargs)
            
            # Decode using appropriate backend
            frames = self._decode_with_backend(video_path, frame_indices, decode_kwargs)
            
            # Cache the decoded frames
            self._cache_frames(cache_key, frames)
            
            # Update statistics
            decode_time = time.time() - start_time
            with self._stats_lock:
                self._stats.total_decodes += 1
                self._stats.total_decode_time += decode_time
                self._stats.total_frames_decoded += len(frames)
                
                decode_time_ms = decode_time * 1000
                if decode_time_ms > self.config.slow_decode_threshold_ms:
                    self._stats.slow_decodes += 1
                    
                    if self.config.log_slow_decodes:
                        warnings.warn(
                            f"Slow video decode: {decode_time_ms:.1f}ms for {len(frame_indices)} "
                            f"frames from {video_path}",
                            UserWarning,
                            stacklevel=2,
                        )
            
            return frames
            
        except Exception as e:
            decode_time = time.time() - start_time
            with self._stats_lock:
                self._stats.total_decodes += 1
                self._stats.total_decode_time += decode_time
            
            raise RuntimeError(f"Video decoding failed for {video_path}: {e}") from e
    
    def _decode_with_backend(
        self,
        video_path: str,
        frame_indices: np.ndarray,
        kwargs: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Decode frames using the configured backend."""
        if self.config.backend == "torchcodec":
            return self._decode_with_torchcodec(video_path, frame_indices, kwargs)
        elif self.config.backend == "opencv":
            return self._decode_with_opencv(video_path, frame_indices, kwargs)
        elif self.config.backend == "decord":
            return self._decode_with_decord(video_path, frame_indices, kwargs)
        else:
            # Fallback to gr00t's video utils if available
            try:
                from gr00t.utils.video_utils import get_frames_by_indices
                return get_frames_by_indices(video_path, frame_indices, **kwargs)
            except ImportError:
                raise ValueError(f"Unsupported video backend: {self.config.backend}")
    
    def _decode_with_torchcodec(
        self,
        video_path: str,
        frame_indices: np.ndarray,
        kwargs: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Decode frames using TorchCodec backend."""
        try:
            import torchcodec
            from torchcodec.decoders import VideoDecoder
            
            # Create decoder with optimized settings
            decoder = VideoDecoder(video_path, **kwargs)
            
            # Decode frames
            frames = []
            for idx in frame_indices:
                frame_tensor = decoder.get_frame_at(idx)
                # Convert to numpy array
                if hasattr(frame_tensor, 'numpy'):
                    frame = frame_tensor.numpy()
                else:
                    frame = np.array(frame_tensor)
                frames.append(frame)
            
            return frames
            
        except ImportError:
            raise ImportError("TorchCodec not available. Install with: pip install torchcodec")
    
    def _decode_with_opencv(
        self,
        video_path: str,
        frame_indices: np.ndarray,
        kwargs: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Decode frames using OpenCV backend."""
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            
            # Set backend if specified
            if "backend" in kwargs:
                backend_name = kwargs["backend"]
                if hasattr(cv2, backend_name):
                    cap.set(cv2.CAP_PROP_BACKEND, getattr(cv2, backend_name))
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    raise RuntimeError(f"Failed to read frame {idx}")
            
            cap.release()
            return frames
            
        except ImportError:
            raise ImportError("OpenCV not available. Install with: pip install opencv-python")
    
    def _decode_with_decord(
        self,
        video_path: str,
        frame_indices: np.ndarray,
        kwargs: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Decode frames using Decord backend."""
        try:
            import decord
            
            # Set context
            ctx = kwargs.get("ctx", "cpu")
            if ctx == "gpu":
                decord.bridge.set_bridge("torch")
            
            # Create video reader
            vr = decord.VideoReader(video_path, ctx=decord.gpu() if ctx == "gpu" else decord.cpu())
            
            # Decode frames
            frames_tensor = vr.get_batch(frame_indices.tolist())
            
            # Convert to numpy
            if hasattr(frames_tensor, 'numpy'):
                frames_array = frames_tensor.numpy()
            else:
                frames_array = np.array(frames_tensor)
            
            # Split into individual frames
            frames = [frames_array[i] for i in range(frames_array.shape[0])]
            
            return frames
            
        except ImportError:
            raise ImportError("Decord not available. Install with: pip install decord")
    
    def prefetch_frames(
        self,
        video_path: str,
        frame_indices: np.ndarray,
        **kwargs
    ):
        """
        Prefetch frames into cache (non-blocking).
        
        Args:
            video_path: Path to video file
            frame_indices: Array of frame indices to prefetch
            **kwargs: Additional arguments for video decoder
        """
        if not self.config.enable_frame_cache:
            return
        
        cache_key = self._cache_key(video_path, frame_indices)
        
        # Check if already cached
        if self._get_cached_frames(cache_key) is not None:
            return
        
        # Prefetch in background thread
        def _prefetch_worker():
            try:
                self.decode_video_frames(video_path, frame_indices, **kwargs)
            except Exception:
                # Ignore prefetch errors to avoid disrupting main flow
                pass
        
        thread = threading.Thread(target=_prefetch_worker, daemon=True)
        thread.start()
    
    def get_stats(self) -> VideoDecodeStats:
        """Get video decoding statistics."""
        with self._stats_lock:
            stats = VideoDecodeStats(
                total_decodes=self._stats.total_decodes,
                total_decode_time=self._stats.total_decode_time,
                total_frames_decoded=self._stats.total_frames_decoded,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                slow_decodes=self._stats.slow_decodes,
            )
            
            # Calculate derived statistics
            if stats.total_decodes > 0:
                stats.average_decode_time_ms = (stats.total_decode_time / stats.total_decodes) * 1000
            
            if stats.total_decode_time > 0:
                stats.average_frames_per_second = stats.total_frames_decoded / stats.total_decode_time
            
            total_cache_requests = stats.cache_hits + stats.cache_misses
            if total_cache_requests > 0:
                stats.cache_hit_rate = stats.cache_hits / total_cache_requests
            
            return stats
    
    def reset_stats(self):
        """Reset video decoding statistics."""
        with self._stats_lock:
            self._stats = VideoDecodeStats()
    
    def clear_cache(self):
        """Clear frame cache."""
        with self._cache_lock:
            self._frame_cache.clear()
            self._current_cache_size = 0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get frame cache information."""
        with self._cache_lock:
            return {
                "cache_size_mb": self._current_cache_size / (1024 * 1024),
                "max_cache_size_mb": self._max_cache_size / (1024 * 1024),
                "cached_items": len(self._frame_cache),
                "cache_utilization": self._current_cache_size / self._max_cache_size if self._max_cache_size > 0 else 0.0,
            }
    
    def optimize_for_dataset(
        self,
        sample_video_paths: List[str],
        sample_frame_counts: List[int],
    ) -> VideoBackendConfig:
        """
        Optimize configuration based on dataset characteristics.
        
        Args:
            sample_video_paths: Sample video paths for benchmarking
            sample_frame_counts: Typical frame counts per video
            
        Returns:
            Optimized video backend configuration
        """
        if not sample_video_paths:
            return self.config
        
        # Benchmark different configurations
        configs_to_test = [
            VideoBackendConfig(backend="torchcodec", num_threads=2),
            VideoBackendConfig(backend="torchcodec", num_threads=4),
            VideoBackendConfig(backend="torchcodec", num_threads=8),
        ]
        
        # Add OpenCV and Decord if available
        try:
            import cv2
            configs_to_test.append(VideoBackendConfig(backend="opencv", num_threads=4))
        except ImportError:
            pass
        
        try:
            import decord
            configs_to_test.append(VideoBackendConfig(backend="decord", num_threads=4))
        except ImportError:
            pass
        
        best_config = self.config
        best_fps = 0.0
        
        for config in configs_to_test:
            try:
                # Create temporary optimizer with test config
                test_optimizer = VideoBackendOptimizer(config)
                
                # Benchmark on sample videos
                start_time = time.time()
                total_frames = 0
                
                for video_path, frame_count in zip(sample_video_paths[:3], sample_frame_counts[:3]):
                    if not os.path.exists(video_path):
                        continue
                    
                    # Test decoding a few frames
                    test_indices = np.arange(min(5, frame_count))
                    frames = test_optimizer.decode_video_frames(video_path, test_indices)
                    total_frames += len(frames)
                
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = total_frames / elapsed_time
                    if fps > best_fps:
                        best_fps = fps
                        best_config = config
                
            except Exception:
                # Skip configurations that fail
                continue
        
        return best_config


# Global video backend optimizer instance
_global_video_optimizer: Optional[VideoBackendOptimizer] = None


def get_global_video_optimizer(config: Optional[VideoBackendConfig] = None) -> VideoBackendOptimizer:
    """
    Get or create global video backend optimizer.
    
    Args:
        config: Video backend configuration (None for auto-detection)
        
    Returns:
        Global video backend optimizer instance
    """
    global _global_video_optimizer
    
    if _global_video_optimizer is None:
        _global_video_optimizer = VideoBackendOptimizer(config)
    
    return _global_video_optimizer


def shutdown_global_video_optimizer():
    """Shutdown global video optimizer."""
    global _global_video_optimizer
    
    if _global_video_optimizer is not None:
        _global_video_optimizer.clear_cache()
        _global_video_optimizer = None


def create_optimized_video_backend_kwargs(
    backend: str = "torchcodec",
    enable_hardware_acceleration: bool = True,
    **extra_kwargs
) -> Dict[str, Any]:
    """
    Create optimized video backend kwargs.
    
    Args:
        backend: Video backend name
        enable_hardware_acceleration: Whether to enable hardware acceleration
        **extra_kwargs: Additional backend-specific arguments
        
    Returns:
        Optimized backend kwargs
    """
    config = VideoBackendConfig(
        backend=backend,
        hardware_acceleration=enable_hardware_acceleration,
        extra_kwargs=extra_kwargs,
    )
    
    optimizer = VideoBackendOptimizer(config)
    return optimizer.config.extra_kwargs