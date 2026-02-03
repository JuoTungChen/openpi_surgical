"""
Parallel action transforms processor for GR00T dataset optimization.

This module provides parallel processing capabilities for action transformations
to improve GPU utilization by reducing CPU bottlenecks in the data pipeline.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError("openpi requires torch for data loading.") from e


@dataclass
class ActionTransformBatch:
    """Batch of action transform requests."""
    actions: List[Dict[str, np.ndarray]]
    states: List[Dict[str, np.ndarray]]
    embodiment_tags: List[str]
    stats_keys: List[str]
    batch_id: int


@dataclass
class ActionTransformResult:
    """Result of parallel action transform processing."""
    processed_actions: List[Dict[str, np.ndarray]]
    processed_states: List[Dict[str, np.ndarray]]
    batch_id: int
    processing_time: float


class ParallelActionProcessor:
    """
    Parallel wrapper for GR00T StateActionProcessor to improve throughput.
    
    This class provides batched and parallel processing of action transformations
    to reduce CPU bottlenecks in the data loading pipeline.
    """
    
    def __init__(
        self,
        modality_configs: Dict[str, Any],
        statistics: Dict[str, Any],
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = True,
        max_workers: Optional[int] = None,
        batch_size: int = 32,
        use_threading: bool = False,
    ):
        """
        Initialize parallel action processor.
        
        Args:
            modality_configs: GR00T modality configurations
            statistics: Normalization statistics
            use_percentiles: Whether to use percentile-based normalization
            clip_outliers: Whether to clip outliers during normalization
            apply_sincos_state_encoding: Whether to apply sin/cos encoding to states
            use_relative_action: Whether to use relative action representations
            max_workers: Maximum number of worker processes/threads (None for auto)
            batch_size: Batch size for parallel processing
            use_threading: Whether to use threads instead of processes
        """
        self.modality_configs = modality_configs
        self.statistics = statistics
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.batch_size = batch_size
        self.use_threading = use_threading
        
        # Determine optimal number of workers
        if max_workers is None:
            if use_threading:
                # For I/O bound tasks, use more threads
                max_workers = min(32, (mp.cpu_count() or 1) * 2)
            else:
                # For CPU bound tasks, use number of CPU cores
                max_workers = max(1, (mp.cpu_count() or 1) - 1)
        
        self.max_workers = max_workers
        self._executor = None
        self._processor_cache = {}
        
        # Performance tracking
        self._total_processed = 0
        self._total_time = 0.0
        self._batch_times = []
    
    def _get_processor(self) -> Any:
        """Get or create StateActionProcessor for current process."""
        process_id = mp.current_process().pid
        
        if process_id not in self._processor_cache:
            try:
                from gr00t.data.state_action.state_action_processor import StateActionProcessor
                
                processor = StateActionProcessor(
                    modality_configs=self.modality_configs,
                    statistics=self.statistics,
                    use_percentiles=self.use_percentiles,
                    clip_outliers=self.clip_outliers,
                    apply_sincos_state_encoding=self.apply_sincos_state_encoding,
                    use_relative_action=self.use_relative_action,
                )
                processor.eval()  # Set to eval mode
                self._processor_cache[process_id] = processor
            except ImportError:
                warnings.warn(
                    "GR00T StateActionProcessor not available. "
                    "Parallel action processing will be disabled.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
        
        return self._processor_cache[process_id]
    
    def _process_single_item(
        self,
        actions: Dict[str, np.ndarray],
        states: Dict[str, np.ndarray],
        embodiment_tag: str,
        stats_key: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Process a single action/state pair."""
        processor = self._get_processor()
        if processor is None:
            # Fallback: return unprocessed data
            return actions, states
        
        try:
            # Apply state processing
            processed_states = processor.apply_state(
                states,
                embodiment_tag=embodiment_tag,
                stats_key=stats_key,
            )
            
            # Apply action processing
            processed_actions = processor.apply_action(
                actions,
                embodiment_tag=embodiment_tag,
                state=states,  # Pass raw states for relative conversion
                stats_key=stats_key,
            )
            
            return processed_actions, processed_states
        except Exception as e:
            warnings.warn(
                f"Action processing failed for embodiment {embodiment_tag}: {e}. "
                f"Returning unprocessed data.",
                UserWarning,
                stacklevel=2,
            )
            return actions, states
    
    def _process_batch_worker(self, batch: ActionTransformBatch) -> ActionTransformResult:
        """Worker function to process a batch of action transforms."""
        import time
        start_time = time.time()
        
        processed_actions = []
        processed_states = []
        
        for actions, states, embodiment_tag, stats_key in zip(
            batch.actions, batch.states, batch.embodiment_tags, batch.stats_keys
        ):
            proc_actions, proc_states = self._process_single_item(
                actions, states, embodiment_tag, stats_key
            )
            processed_actions.append(proc_actions)
            processed_states.append(proc_states)
        
        processing_time = time.time() - start_time
        
        return ActionTransformResult(
            processed_actions=processed_actions,
            processed_states=processed_states,
            batch_id=batch.batch_id,
            processing_time=processing_time,
        )
    
    def process_batch(
        self,
        actions_list: List[Dict[str, np.ndarray]],
        states_list: List[Dict[str, np.ndarray]],
        embodiment_tags: List[str],
        stats_keys: List[str],
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        """
        Process a batch of action transforms in parallel.
        
        Args:
            actions_list: List of action dictionaries
            states_list: List of state dictionaries
            embodiment_tags: List of embodiment tag strings
            stats_keys: List of statistics keys
            
        Returns:
            Tuple of (processed_actions_list, processed_states_list)
        """
        if len(actions_list) != len(states_list) != len(embodiment_tags) != len(stats_keys):
            raise ValueError("All input lists must have the same length")
        
        if len(actions_list) == 0:
            return [], []
        
        # For small batches, process synchronously to avoid overhead
        if len(actions_list) <= 4 or self.max_workers == 1:
            processed_actions = []
            processed_states = []
            
            for actions, states, embodiment_tag, stats_key in zip(
                actions_list, states_list, embodiment_tags, stats_keys
            ):
                proc_actions, proc_states = self._process_single_item(
                    actions, states, embodiment_tag, stats_key
                )
                processed_actions.append(proc_actions)
                processed_states.append(proc_states)
            
            return processed_actions, processed_states
        
        # Create batches for parallel processing
        batches = []
        for i in range(0, len(actions_list), self.batch_size):
            end_idx = min(i + self.batch_size, len(actions_list))
            batch = ActionTransformBatch(
                actions=actions_list[i:end_idx],
                states=states_list[i:end_idx],
                embodiment_tags=embodiment_tags[i:end_idx],
                stats_keys=stats_keys[i:end_idx],
                batch_id=i // self.batch_size,
            )
            batches.append(batch)
        
        # Process batches in parallel
        if self._executor is None:
            executor_class = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
            self._executor = executor_class(max_workers=self.max_workers)
        
        try:
            # Submit all batches
            future_to_batch = {
                self._executor.submit(self._process_batch_worker, batch): batch
                for batch in batches
            }
            
            # Collect results in order
            results = [None] * len(batches)
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results[result.batch_id] = result
                    
                    # Update performance tracking
                    self._total_processed += len(batch.actions)
                    self._total_time += result.processing_time
                    self._batch_times.append(result.processing_time)
                    
                except Exception as e:
                    warnings.warn(
                        f"Batch {batch.batch_id} processing failed: {e}. "
                        f"Using fallback processing.",
                        UserWarning,
                        stacklevel=2,
                    )
                    # Fallback: process synchronously
                    fallback_result = self._process_batch_worker(batch)
                    results[fallback_result.batch_id] = fallback_result
            
            # Flatten results
            processed_actions = []
            processed_states = []
            
            for result in results:
                if result is not None:
                    processed_actions.extend(result.processed_actions)
                    processed_states.extend(result.processed_states)
            
            return processed_actions, processed_states
            
        except Exception as e:
            warnings.warn(
                f"Parallel processing failed: {e}. Falling back to synchronous processing.",
                UserWarning,
                stacklevel=2,
            )
            # Fallback to synchronous processing
            processed_actions = []
            processed_states = []
            
            for actions, states, embodiment_tag, stats_key in zip(
                actions_list, states_list, embodiment_tags, stats_keys
            ):
                proc_actions, proc_states = self._process_single_item(
                    actions, states, embodiment_tag, stats_key
                )
                processed_actions.append(proc_actions)
                processed_states.append(proc_states)
            
            return processed_actions, processed_states
    
    def process_single(
        self,
        actions: Dict[str, np.ndarray],
        states: Dict[str, np.ndarray],
        embodiment_tag: str,
        stats_key: str,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Process a single action/state pair.
        
        Args:
            actions: Action dictionary
            states: State dictionary
            embodiment_tag: Embodiment tag string
            stats_key: Statistics key
            
        Returns:
            Tuple of (processed_actions, processed_states)
        """
        return self._process_single_item(actions, states, embodiment_tag, stats_key)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the processor."""
        if self._total_processed == 0:
            return {
                "total_processed": 0,
                "average_time_per_item": 0.0,
                "total_time": 0.0,
                "throughput_items_per_sec": 0.0,
                "batch_count": 0,
                "average_batch_time": 0.0,
            }
        
        avg_time_per_item = self._total_time / self._total_processed
        throughput = self._total_processed / self._total_time if self._total_time > 0 else 0.0
        avg_batch_time = np.mean(self._batch_times) if self._batch_times else 0.0
        
        return {
            "total_processed": self._total_processed,
            "average_time_per_item": avg_time_per_item,
            "total_time": self._total_time,
            "throughput_items_per_sec": throughput,
            "batch_count": len(self._batch_times),
            "average_batch_time": avg_batch_time,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "use_threading": self.use_threading,
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._total_processed = 0
        self._total_time = 0.0
        self._batch_times = []
    
    def shutdown(self):
        """Shutdown the executor and clean up resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        
        # Clear processor cache
        self._processor_cache.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


# Global processor instance for reuse across dataset workers
_global_parallel_processor: Optional[ParallelActionProcessor] = None


def get_global_parallel_processor(
    modality_configs: Dict[str, Any],
    statistics: Dict[str, Any],
    **kwargs
) -> ParallelActionProcessor:
    """
    Get or create a global parallel action processor instance.
    
    This function provides a singleton pattern for the processor to avoid
    creating multiple instances across different dataset workers.
    """
    global _global_parallel_processor
    
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            **kwargs
        )
    
    return _global_parallel_processor


def shutdown_global_processor():
    """Shutdown the global parallel processor."""
    global _global_parallel_processor
    
    if _global_parallel_processor is not None:
        _global_parallel_processor.shutdown()
        _global_parallel_processor = None