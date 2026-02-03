"""
Async video decoder for GR00T datasets to improve GPU utilization.

This module provides background video decoding capabilities to prevent video I/O
from blocking GPU computation during training.
"""

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import weakref

import numpy as np

try:
    from gr00t.utils.video_utils import get_frames_by_indices
except ImportError:
    # Fallback for when gr00t is not available
    def get_frames_by_indices(*args, **kwargs):
        raise ImportError("gr00t is required for video decoding")


@dataclass
class VideoDecodeRequest:
    """Request for video frame decoding."""
    video_path: str
    frame_indices: np.ndarray
    video_backend: str = "torchcodec"
    video_backend_kwargs: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


@dataclass
class VideoDecodeResult:
    """Result of video frame decoding."""
    frames: List[np.ndarray]
    request_id: Optional[str] = None
    decode_time: float = 0.0
    cache_hit: bool = False


class VideoFrameCache:
    """LRU cache for decoded video frames with memory management."""
    
    def __init__(self, max_size_mb: int = 512, max_entries: int = 1000):
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self._cache = {}
        self._access_times = {}
        self._current_size_mb = 0.0
        self._lock = threading.RLock()
        
    def _estimate_frame_size_mb(self, frame: np.ndarray) -> float:
        """Estimate memory size of a frame in MB."""
        return frame.nbytes / (1024 * 1024)
    
    def _evict_lru(self):
        """Evict least recently used entries to free memory."""
        if not self._cache:
            return
            
        # Sort by access time and remove oldest entries
        sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        for key in sorted_keys:
            if (self._current_size_mb <= self.max_size_mb and 
                len(self._cache) <= self.max_entries):
                break
                
            frames = self._cache.pop(key, None)
            if frames:
                for frame in frames:
                    self._current_size_mb -= self._estimate_frame_size_mb(frame)
                del self._access_times[key]
    
    def get(self, key: str) -> Optional[List[np.ndarray]]:
        """Get cached frames if available."""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key].copy()  # Return copy to avoid modification
            return None
    
    def put(self, key: str, frames: List[np.ndarray]):
        """Cache decoded frames."""
        with self._lock:
            # Calculate size of new frames
            new_size = sum(self._estimate_frame_size_mb(frame) for frame in frames)
            
            # Don't cache if single request is too large
            if new_size > self.max_size_mb * 0.5:
                return
            
            # Evict old entries if needed
            self._current_size_mb += new_size
            self._evict_lru()
            
            # Store frames (make copies to avoid external modification)
            self._cache[key] = [frame.copy() for frame in frames]
            self._access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached frames."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._current_size_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "size_mb": self._current_size_mb,
                "max_size_mb": self.max_size_mb,
                "max_entries": self.max_entries,
            }


class AsyncVideoDecoder:
    """Asynchronous video decoder with caching and background processing."""
    
    def __init__(
        self,
        max_workers: int = 4,
        cache_size_mb: int = 512,
        cache_entries: int = 1000,
        prefetch_queue_size: int = 32,
    ):
        """Initialize async video decoder.
        
        Args:
            max_workers: Maximum number of worker threads for video decoding
            cache_size_mb: Maximum cache size in MB
            cache_entries: Maximum number of cached entries
            prefetch_queue_size: Maximum size of prefetch request queue
        """
        self.max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._cache = VideoFrameCache(cache_size_mb, cache_entries)
        self._prefetch_queue = asyncio.Queue(maxsize=prefetch_queue_size)
        self._prefetch_task = None
        self._shutdown = False
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "decode_time_total": 0.0,
            "decode_time_avg": 0.0,
        }
        self._stats_lock = threading.Lock()
        
        # Start prefetch worker
        self._start_prefetch_worker()
    
    def _start_prefetch_worker(self):
        """Start background prefetch worker."""
        def run_prefetch_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._prefetch_worker())
            loop.close()
        
        self._prefetch_thread = threading.Thread(target=run_prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    async def _prefetch_worker(self):
        """Background worker for prefetching video frames."""
        while not self._shutdown:
            try:
                # Wait for prefetch requests with timeout
                request = await asyncio.wait_for(
                    self._prefetch_queue.get(), timeout=1.0
                )
                
                # Decode in background (don't wait for result)
                asyncio.create_task(self._decode_async(request))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.warning(f"Prefetch worker error: {e}")
    
    def _make_cache_key(self, request: VideoDecodeRequest) -> str:
        """Create cache key for video decode request."""
        indices_str = ",".join(map(str, request.frame_indices))
        return f"{request.video_path}:{indices_str}:{request.video_backend}"
    
    def _decode_sync(self, request: VideoDecodeRequest) -> VideoDecodeResult:
        """Synchronously decode video frames."""
        start_time = time.time()
        
        try:
            frames = get_frames_by_indices(
                request.video_path,
                request.frame_indices,
                video_backend=request.video_backend,
                video_backend_kwargs=request.video_backend_kwargs or {},
            )
            
            # Convert to numpy arrays if needed
            frames = [np.asarray(frame) for frame in frames]
            
            decode_time = time.time() - start_time
            
            return VideoDecodeResult(
                frames=frames,
                request_id=request.request_id,
                decode_time=decode_time,
                cache_hit=False,
            )
            
        except Exception as e:
            logging.error(f"Video decode error for {request.video_path}: {e}")
            # Return empty frames on error
            return VideoDecodeResult(
                frames=[],
                request_id=request.request_id,
                decode_time=time.time() - start_time,
                cache_hit=False,
            )
    
    async def _decode_async(self, request: VideoDecodeRequest) -> VideoDecodeResult:
        """Asynchronously decode video frames."""
        cache_key = self._make_cache_key(request)
        
        # Check cache first
        cached_frames = self._cache.get(cache_key)
        if cached_frames is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            
            return VideoDecodeResult(
                frames=cached_frames,
                request_id=request.request_id,
                decode_time=0.0,
                cache_hit=True,
            )
        
        # Decode in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, self._decode_sync, request)
        
        # Cache successful results
        if result.frames:
            self._cache.put(cache_key, result.frames)
        
        # Update statistics
        with self._stats_lock:
            self._stats["total_requests"] += 1
            self._stats["cache_misses"] += 1
            self._stats["decode_time_total"] += result.decode_time
            if self._stats["total_requests"] > 0:
                self._stats["decode_time_avg"] = (
                    self._stats["decode_time_total"] / self._stats["total_requests"]
                )
        
        return result
    
    def decode_sync(self, request: VideoDecodeRequest) -> VideoDecodeResult:
        """Synchronously decode video frames (blocking call)."""
        cache_key = self._make_cache_key(request)
        
        # Check cache first
        cached_frames = self._cache.get(cache_key)
        if cached_frames is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            
            return VideoDecodeResult(
                frames=cached_frames,
                request_id=request.request_id,
                decode_time=0.0,
                cache_hit=True,
            )
        
        # Decode synchronously
        result = self._decode_sync(request)
        
        # Cache successful results
        if result.frames:
            self._cache.put(cache_key, result.frames)
        
        # Update statistics
        with self._stats_lock:
            self._stats["total_requests"] += 1
            self._stats["cache_misses"] += 1
            self._stats["decode_time_total"] += result.decode_time
            if self._stats["total_requests"] > 0:
                self._stats["decode_time_avg"] = (
                    self._stats["decode_time_total"] / self._stats["total_requests"]
                )
        
        return result
    
    def prefetch(self, request: VideoDecodeRequest):
        """Queue a video decode request for background prefetching."""
        if self._shutdown:
            return
        
        try:
            # Non-blocking put - drop request if queue is full
            self._prefetch_queue.put_nowait(request)
        except asyncio.QueueFull:
            logging.debug("Prefetch queue full, dropping request")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decoder statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
        
        cache_stats = self._cache.get_stats()
        stats.update({
            "cache_stats": cache_stats,
            "max_workers": self.max_workers,
            "cache_hit_rate": (
                stats["cache_hits"] / max(1, stats["total_requests"])
            ),
        })
        
        return stats
    
    def clear_cache(self):
        """Clear video frame cache."""
        self._cache.clear()
    
    def shutdown(self):
        """Shutdown the decoder and cleanup resources."""
        self._shutdown = True
        
        # Wait for prefetch thread to finish
        if hasattr(self, '_prefetch_thread'):
            self._prefetch_thread.join(timeout=2.0)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear cache
        self._cache.clear()


# Global decoder instance (lazy initialization)
_global_decoder: Optional[AsyncVideoDecoder] = None
_decoder_lock = threading.Lock()


def get_global_decoder() -> AsyncVideoDecoder:
    """Get or create global async video decoder instance."""
    global _global_decoder
    
    with _decoder_lock:
        if _global_decoder is None:
            _global_decoder = AsyncVideoDecoder()
        return _global_decoder


def shutdown_global_decoder():
    """Shutdown global decoder instance."""
    global _global_decoder
    
    with _decoder_lock:
        if _global_decoder is not None:
            _global_decoder.shutdown()
            _global_decoder = None


# Cleanup on module exit
import atexit
atexit.register(shutdown_global_decoder)