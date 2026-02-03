"""
Adapter dataset to reuse `gr00t` LeRobot loaders inside `openpi`.

Goal: allow `openpi` training to *import* and use the data loading + embodiment modality
configs from `gr00t` without copying code.

This module provides a torch-style random-access Dataset that yields per-timestep samples
with an action chunk (horizon) in the same *flat-key* style as LeRobot datasets used by
`openpi.training.data_loader`:

  - "observation.images.<view>" : uint8[H,W,3]
  - "observation.state"         : float32[state_dim]
  - "<action_key>" (default "actions") : float32[action_horizon, action_dim]
  - "prompt"                    : str (numpy scalar string) if available

The rest of the `openpi` pipeline (repack transforms, normalization, tokenization, etc.)
remains unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import warnings

import numpy as np
import pandas as pd

try:
    import torch
except Exception as e:  # pragma: no cover
    raise ImportError("openpi requires torch for data loading.") from e

# Import async video decoder, parallel action processor, intelligent caching, and video optimization
from openpi.training.async_video_decoder import (
    AsyncVideoDecoder,
    VideoDecodeRequest,
    get_global_decoder,
)
from openpi.training.parallel_action_processor import (
    ParallelActionProcessor,
    get_global_parallel_processor,
)
from openpi.training.intelligent_episode_cache import (
    DistributedEpisodeCache,
    get_global_episode_cache,
    create_cached_loader,
)
from openpi.training.video_backend_optimizer import (
    VideoBackendOptimizer,
    VideoBackendConfig,
    get_global_video_optimizer,
    create_optimized_video_backend_kwargs,
)


def _require_gr00t() -> None:
    try:
        import gr00t  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import `gr00t`.\n"
            "To use the gr00t-backed dataset in openpi, install gr00t into the same environment, e.g.:\n"
            "  `pip install -e /home/iulian/chole_ws/src/gr00t_n1.6`\n"
            "or add it to PYTHONPATH.\n"
        ) from e


@dataclass(frozen=True)
class Gr00tDatasetSpec:
    dataset_path: str
    embodiment_tag: str
    # Optional path to GR00T modality config file (e.g., dVRK_config.py).
    # If provided, will be imported to register the embodiment with MODALITY_CONFIGS.
    modality_config_path: str | None = None
    # Action horizon for action chunking when we cannot rely on a gr00t embodiment config.
    action_horizon: int = 16
    # If provided, restrict to this subset of episodes (episode indices in the dataset).
    episode_indices: np.ndarray | None = None
    # Which language field to map into "prompt". Typical values: "task", "sub_task", or
    # full annotation keys like "annotation.human.action.task_description" depending on dataset.
    language_key: str | None = None
    # Which video view keys to expose as observation images. If None, uses gr00t modality config order.
    video_views: list[str] | None = None
    # Cache size for decoded episodes per worker process.
    # Larger values reduce video decoding overhead but use more memory.
    # Recommended: Set to number of episodes / num_workers for best performance.
    episode_cache_size: int = 16
    # Intelligent caching settings
    enable_intelligent_caching: bool = True
    cache_memory_mb_per_worker: int = 512
    enable_cross_process_cache_sharing: bool = True
    enable_cache_warming: bool = True
    # Video backend args forwarded into gr00t loader.
    video_backend: str = "torchcodec"
    video_backend_kwargs: dict[str, Any] | None = None
    # If True, apply GR00T's action representation transformations (e.g., hybrid-relative, rot6d).
    # When True: actions are processed according to ActionConfig (e.g., 20D for dVRK with hybrid-relative)
    # When False: actions remain in raw format from dataset (e.g., 16D for dVRK with xyz+quat)
    apply_action_transforms: bool = True
    # Statistics key for normalization (if None, will be inferred from embodiment_tag).
    # Only used if apply_action_transforms=True.
    stats_key: str | None = None
    # Async video decoding settings
    enable_async_video_decoding: bool = True
    video_cache_size_mb: int = 512
    video_prefetch_workers: int = 4
    # Parallel action processing settings
    enable_parallel_action_processing: bool = True
    action_processing_workers: int = None  # None for auto-detection
    action_processing_batch_size: int = 32
    use_threading_for_actions: bool = False
    # Video backend optimization settings
    enable_video_backend_optimization: bool = True
    video_backend_config: Optional[VideoBackendConfig] = None
    optimize_video_for_dataset: bool = True


class Gr00tLeRobotTorchDataset(torch.utils.data.Dataset):
    """
    Random-access dataset backed by gr00t's LeRobotEpisodeLoader + modality configs.

    Each element corresponds to a (episode_id, step_index) pair, where step_index is a
    valid base index for the configured action horizon (i.e., we don't sample beyond
    the end of an episode).
    """

    def __init__(self, spec: Gr00tDatasetSpec):
        _require_gr00t()

        self._spec = spec
        
        # Initialize critical attributes first to avoid AttributeError
        self._total_steps = 0
        self._episode_ids = np.array([], dtype=np.int32)
        self._effective_lengths = np.array([], dtype=np.int64)
        self._cum_lengths = np.array([], dtype=np.int64)
        self._video_views = []
        self._video_path_pattern = None
        
        # Initialize optimization components as None - they will be lazy-initialized
        # to avoid pickle issues with multiprocessing DataLoader
        self._async_video_decoder = None
        self._parallel_action_processor = None
        self._distributed_cache = None
        self._video_optimizer = None
        
        # Store configuration for lazy initialization (avoid storing complex objects)
        self._async_video_config = {
            'enabled': spec.enable_async_video_decoding,
            'max_workers': spec.video_prefetch_workers,
            'cache_size_mb': spec.video_cache_size_mb,
        }
        
        # Store video optimizer config as simple dict to avoid pickle issues
        self._video_optimizer_config = {
            'enabled': spec.enable_video_backend_optimization,
            'backend': spec.video_backend,
            'extra_kwargs': spec.video_backend_kwargs or {},
            'optimize_for_dataset': spec.optimize_video_for_dataset,
        }
        
        # Initialize the dataset
        self._initialize_dataset()

    def _get_async_video_decoder(self):
        """Lazy initialization of async video decoder to avoid pickle issues."""
        if self._async_video_decoder is None and self._async_video_config['enabled']:
            from openpi.training.async_video_decoder import AsyncVideoDecoder
            self._async_video_decoder = AsyncVideoDecoder(
                max_workers=self._async_video_config['max_workers'],
                cache_size_mb=self._async_video_config['cache_size_mb'],
            )
        return self._async_video_decoder
    
    def _get_video_optimizer(self):
        """Lazy initialization of video optimizer to avoid pickle issues."""
        if self._video_optimizer is None and self._video_optimizer_config['enabled']:
            from openpi.training.video_backend_optimizer import get_global_video_optimizer, VideoBackendConfig
            config = VideoBackendConfig(
                backend=self._video_optimizer_config['backend'],
                extra_kwargs=self._video_optimizer_config['extra_kwargs'],
            )
            self._video_optimizer = get_global_video_optimizer(config)
        return self._video_optimizer
    
    def _get_parallel_action_processor(self):
        """Lazy initialization of parallel action processor to avoid pickle issues."""
        if self._parallel_action_processor is None and hasattr(self._spec, 'enable_parallel_action_processing') and self._spec.enable_parallel_action_processing:
            from openpi.training.parallel_action_processor import get_global_parallel_processor
            self._parallel_action_processor = get_global_parallel_processor(
                modality_configs={self._spec.embodiment_tag: self._modality_configs},
                statistics=self._statistics,
                use_percentiles=True,
                clip_outliers=True,
                apply_sincos_state_encoding=False,
                use_relative_action=True,
                max_workers=self._spec.action_processing_workers,
                batch_size=self._spec.action_processing_batch_size,
                use_threading=self._spec.use_threading_for_actions,
            )
        return self._parallel_action_processor
    
    def _initialize_dataset(self):
        """Initialize the dataset after all lazy components are set up."""
        spec = self._spec
        
        # Import here to avoid storing classes as instance variables
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        
        # Disable advanced optimizations to avoid pickle issues for now
        if spec.enable_intelligent_caching:
            # Skip intelligent caching to avoid pickle issues - it contains thread locks
            pass
            
        # Initialize video backend optimizer if enabled
        self._video_optimizer = None
        # Video optimizer will be lazy-initialized to avoid pickle issues
        # Configuration is stored in self._video_optimizer_config
        
        # If modality_config_path is provided, load and register it
        if spec.modality_config_path:
            from pathlib import Path
            import importlib.util
            
            # Check if already registered to avoid double-registration
            if spec.embodiment_tag not in MODALITY_CONFIGS:
                # Load the modality config module to trigger registration
                config_path = Path(spec.modality_config_path)
                
                # Validate that the config file exists
                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Modality config file not found: {config_path}\n"
                        f"Please provide a valid path via --data.modality-config-path or update the config.\n"
                        f"Expected a Python file that registers the embodiment in MODALITY_CONFIGS."
                    )
                
                try:
                    spec_module = importlib.util.spec_from_file_location("temp_modality_config", config_path)
                    if spec_module and spec_module.loader:
                        module = importlib.util.module_from_spec(spec_module)
                        spec_module.loader.exec_module(module)
                except Exception as e:
                    raise RuntimeError(f"Error loading modality config from {config_path}: {e}")

        self._use_gr00t_modality_config = spec.embodiment_tag in MODALITY_CONFIGS

        if self._use_gr00t_modality_config:
            # --- Standard path: use gr00t configs + LeRobotEpisodeLoader + extract_step_data ---
            from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
            from gr00t.data.embodiment_tags import EmbodimentTag

            self._extract_step_data = extract_step_data
            self._embodiment_tag_class = EmbodimentTag

            self._modality_configs = MODALITY_CONFIGS[spec.embodiment_tag]
            if "action" not in self._modality_configs:
                raise ValueError("gr00t modality config must include an 'action' modality.")
            if "state" not in self._modality_configs:
                raise ValueError("gr00t modality config must include a 'state' modality.")

            self._action_delta_indices = list(self._modality_configs["action"].delta_indices)
            if len(self._action_delta_indices) == 0:
                raise ValueError("action.delta_indices must be non-empty.")
            self._max_action_delta = int(max(self._action_delta_indices))

            # Determine which state index corresponds to "current" timestep (delta == 0).
            state_deltas = list(self._modality_configs["state"].delta_indices)
            if 0 in state_deltas:
                self._state_zero_index = state_deltas.index(0)
            else:
                warnings.warn(
                    f"State modality config for {spec.embodiment_tag} does not include "
                    f"delta=0 (current timestep). Using last delta index ({state_deltas[-1]}) "
                    "as current state. This may cause misalignment between state and actions.",
                    UserWarning,
                    stacklevel=2,
                )
                self._state_zero_index = len(state_deltas) - 1

            # Load statistics and initialize processor if action transforms are requested
            self._stats_key = None
            self._statistics = None
            self._processor = None
            if spec.apply_action_transforms:
                import json
                from pathlib import Path
                from gr00t.data.state_action.state_action_processor import StateActionProcessor

                # GR00T-style behavior: stats are per-dataset and keyed by repo_id at runtime.
                # We always derive stats_key from the dataset path name for training.
                self._stats_key = Path(spec.dataset_path).name
                if spec.stats_key is not None and spec.stats_key != self._stats_key:
                    warnings.warn(
                        f"Ignoring provided stats_key '{spec.stats_key}' and using repo_id '{self._stats_key}' "
                        "to match GR00T per-dataset stats behavior.",
                        UserWarning,
                        stacklevel=2,
                    )

                stats_path = Path(spec.dataset_path) / "meta" / "percentile_stats.json"
                if stats_path.exists():
                    with open(stats_path, "r") as f:
                        stats_data = json.load(f)

                    # If stats are unkeyed (GR00T default), wrap them under stats_key and attach embodiment metadata.
                    # This mirrors gr00t.data.dataset.ShardedMixtureDataset.setup_per_dataset_statistics.
                    if "state" in stats_data and "action" in stats_data:
                        stats_data["__embodiment_tag__"] = spec.embodiment_tag
                        self._statistics = {self._stats_key: stats_data}
                    else:
                        # Already keyed (e.g., consolidated stats file).
                        self._statistics = stats_data

                    if self._stats_key not in self._statistics:
                        warnings.warn(
                            f"Statistics key '{self._stats_key}' not found in {stats_path}. "
                            f"Available keys: {list(self._statistics.keys())}. "
                            f"Action transforms will not be applied.",
                            UserWarning,
                            stacklevel=2,
                        )
                        self._stats_key = None
                        self._statistics = None
                    else:
                        # Initialize StateActionProcessor with loaded statistics.
                        self._processor = StateActionProcessor(
                            modality_configs={spec.embodiment_tag: self._modality_configs},
                            statistics=self._statistics,
                            use_percentiles=True,  # Use percentile-based normalization
                            clip_outliers=True,
                            apply_sincos_state_encoding=False,
                            use_relative_action=True,  # Enable relative/hybrid-relative conversions
                        )
                        self._processor.eval()  # Set to eval mode (no data augmentation)
                        
                        # Initialize parallel action processor if enabled
                        if spec.enable_parallel_action_processing:
                            self._parallel_action_processor = get_global_parallel_processor(
                                modality_configs={spec.embodiment_tag: self._modality_configs},
                                statistics=self._statistics,
                                use_percentiles=True,
                                clip_outliers=True,
                                apply_sincos_state_encoding=False,
                                use_relative_action=True,
                                max_workers=spec.action_processing_workers,
                                batch_size=spec.action_processing_batch_size,
                                use_threading=spec.use_threading_for_actions,
                            )
                else:
                    warnings.warn(
                        f"Statistics file not found at {stats_path}. "
                        f"Action representation transforms (hybrid-relative, rot6d, etc.) will not be applied. "
                        f"Actions will remain in raw format.",
                        UserWarning,
                        stacklevel=2,
                    )

            self._episode_loader = LeRobotEpisodeLoader(
                dataset_path=spec.dataset_path,
                modality_configs=self._modality_configs,
                video_backend=spec.video_backend,
                video_backend_kwargs=spec.video_backend_kwargs,
                skip_video=False,
                require_stats=False,
            )

            if spec.episode_indices is None:
                self._episode_ids = np.arange(len(self._episode_loader), dtype=np.int32)
            else:
                self._episode_ids = np.asarray(spec.episode_indices, dtype=np.int32)

            raw_lengths = np.asarray(
                [self._episode_loader.get_episode_length(int(i)) for i in self._episode_ids], dtype=np.int64
            )
            self._raw_lengths = raw_lengths  # Store for debugging
            effective = np.maximum(raw_lengths - self._max_action_delta, 0)
            self._effective_lengths = effective
            self._cum_lengths = np.cumsum(self._effective_lengths, dtype=np.int64)
            self._total_steps = int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

            if "video" in self._modality_configs:
                self._video_views = spec.video_views or list(self._modality_configs["video"].modality_keys)
            else:
                self._video_views = spec.video_views or []

            # Initialize video path pattern for gr00t modality config path
            # This is needed for video backend optimization
            self._video_path_pattern = None
            if self._video_views:
                # For gr00t datasets, try to infer video path pattern from dataset structure
                # Common patterns: "videos/{episode_chunk:06d}/{video_key}_{episode_index:06d}.mp4"
                self._video_path_pattern = "videos/{episode_chunk:06d}/{video_key}_{episode_index:06d}.mp4"
                
                # Try to read from info.json if available
                try:
                    import json
                    from pathlib import Path
                    info_path = Path(spec.dataset_path) / "meta" / "info.json"
                    if info_path.exists():
                        info_meta = json.loads(info_path.read_text())
                        if "video_path" in info_meta:
                            self._video_path_pattern = info_meta["video_path"]
                        # Also get chunk size if available
                        if "chunks_size" in info_meta:
                            self._chunk_size = int(info_meta["chunks_size"])
                        else:
                            self._chunk_size = 1000  # Default chunk size
                    else:
                        self._chunk_size = 1000  # Default chunk size
                        
                    # Also need dataset path for video optimization
                    self._dataset_path = Path(spec.dataset_path)
                        
                except Exception:
                    # If we can't read info.json, use defaults
                    self._chunk_size = 1000
                    self._dataset_path = Path(spec.dataset_path)

            self._state_keys = list(self._modality_configs["state"].modality_keys)
            self._action_keys = list(self._modality_configs["action"].modality_keys)
        else:
            # --- Fallback path: dataset has LeRobot info.json but no gr00t embodiment config ---
            # This is common for Open-H datasets where robot_type is present (e.g., "dvrk") but
            # the local gr00t checkout may not have that entry yet.
            # We load parquet columns directly and decode videos using gr00t's video utils.
            import json
            from pathlib import Path

            info_path = Path(spec.dataset_path) / "meta" / "info.json"
            episodes_path = Path(spec.dataset_path) / "meta" / "episodes.jsonl"
            if not info_path.exists():
                raise FileNotFoundError(f"Missing {info_path}")
            if not episodes_path.exists():
                raise FileNotFoundError(f"Missing {episodes_path}")

            self._dataset_path = Path(spec.dataset_path)
            self._info_meta = json.loads(info_path.read_text())
            self._data_path_pattern = self._info_meta["data_path"]
            self._video_path_pattern = self._info_meta.get("video_path")
            self._chunk_size = int(self._info_meta["chunks_size"])

            # Parse episode lengths from episodes.jsonl.
            episodes = [json.loads(l) for l in episodes_path.read_text().splitlines() if l.strip()]
            self._episodes_metadata = episodes
            all_episode_ids = np.asarray([int(e["episode_index"]) for e in episodes], dtype=np.int32)

            if spec.episode_indices is None:
                self._episode_ids = all_episode_ids
            else:
                self._episode_ids = np.asarray(spec.episode_indices, dtype=np.int32)

            lengths_by_id = {int(e["episode_index"]): int(e["length"]) for e in episodes}
            raw_lengths = np.asarray([lengths_by_id[int(i)] for i in self._episode_ids], dtype=np.int64)
            self._raw_lengths = raw_lengths  # Store for debugging

            self._action_delta_indices = list(range(int(spec.action_horizon)))
            self._max_action_delta = int(max(self._action_delta_indices)) if self._action_delta_indices else 0

            effective = np.maximum(raw_lengths - self._max_action_delta, 0)
            self._effective_lengths = effective
            self._cum_lengths = np.cumsum(self._effective_lengths, dtype=np.int64)
            self._total_steps = int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

            # Views come from spec (required for fallback).
            self._video_views = spec.video_views or []
            
            # Initialize video path pattern if not present in metadata
            if self._video_path_pattern is None and self._video_views:
                # Try to infer video path pattern from data structure
                # Common patterns: "videos/{episode_chunk:06d}/{video_key}_{episode_index:06d}.mp4"
                self._video_path_pattern = "videos/{episode_chunk:06d}/{video_key}_{episode_index:06d}.mp4"
                logging.warning(f"Video path pattern not found in metadata, using default: {self._video_path_pattern}")

            # State and action come from raw columns.
            self._state_keys = []
            self._action_keys = []

        if self._total_steps <= 0:
            # Provide detailed debug information
            debug_info = []
            debug_info.append(f"Dataset path: {spec.dataset_path}")
            debug_info.append(f"Embodiment tag: {spec.embodiment_tag}")
            debug_info.append(f"Use gr00t modality config: {self._use_gr00t_modality_config}")
            
            if hasattr(self, '_episode_loader'):
                debug_info.append(f"Episode loader length: {len(self._episode_loader)}")
            if hasattr(self, '_episode_ids'):
                debug_info.append(f"Episode IDs: {self._episode_ids}")
            if hasattr(self, '_max_action_delta'):
                debug_info.append(f"Max action delta: {self._max_action_delta}")
            if hasattr(self, '_effective_lengths'):
                debug_info.append(f"Effective lengths: {self._effective_lengths}")
                if len(self._effective_lengths) > 0:
                    debug_info.append(f"Raw lengths before action delta: {getattr(self, '_raw_lengths', 'unknown')}")
            
            debug_msg = "\n".join(debug_info)
            raise ValueError(
                f"No valid (episode, step) pairs found. "
                f"Check your dataset path and that episodes are longer than the action horizon.\n"
                f"Debug information:\n{debug_msg}"
            )
        
        # Warm up cache with first few episodes if enabled
        if (self._distributed_cache is not None and 
            self._spec.enable_cache_warming and 
            len(self._episode_ids) > 0):
            # Prefetch first 10% of episodes or up to 20 episodes
            warmup_count = min(20, max(1, len(self._episode_ids) // 10))
            warmup_episodes = self._episode_ids[:warmup_count].tolist()
            self.prefetch_episodes(warmup_episodes)
        
        # Optimize video backend for dataset if enabled
        if self._spec.optimize_video_for_dataset:
            self.optimize_video_backend_for_dataset()

    def __len__(self) -> int:
        return self._total_steps

    def get_consolidated_statistics(self) -> dict[str, dict[str, Any]] | None:
        """Return GR00T-style keyed statistics for this dataset.

        The returned dict is keyed by `stats_key` (repo_id), matching GR00T's
        `use_per_dataset_stats` flow. Each entry includes `__embodiment_tag__`
        so downstream components can look up the correct modality config.

        Returns:
            The keyed statistics dict if available, otherwise None.
        """
        return self._statistics

    def _global_to_episode_step(self, index: int) -> tuple[int, int]:
        if index < 0:
            index = self._total_steps + index
        if index < 0 or index >= self._total_steps:
            raise IndexError(index)

        ep_pos = int(np.searchsorted(self._cum_lengths, index, side="right"))
        prev = int(self._cum_lengths[ep_pos - 1]) if ep_pos > 0 else 0
        step = int(index - prev)
        episode_id = int(self._episode_ids[ep_pos])
        return episode_id, step

    def _concat_state(self, vla_states: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for k in self._state_keys:
            arr = np.asarray(vla_states[k], dtype=np.float32)
            if arr.ndim == 1:
                parts.append(arr)
            else:
                parts.append(arr[self._state_zero_index])
        return np.concatenate(parts, axis=-1) if parts else np.zeros((0,), dtype=np.float32)

    def _concat_actions(self, vla_actions: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for k in self._action_keys:
            arr = np.asarray(vla_actions[k], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected action[{k}] to be (horizon, dim), got shape {arr.shape}")
            parts.append(arr)
        if not parts:
            return np.zeros((len(self._action_delta_indices), 0), dtype=np.float32)
        # Concatenate along last dim -> (horizon, total_dim)
        return np.concatenate(parts, axis=-1)

    def _select_images(self, vla_images: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for view in self._video_views:
            frames = vla_images.get(view)
            if frames is None or len(frames) == 0:
                continue
            # gr00t stores a list (temporal stack). For openpi we keep the "current" frame.
            img = np.asarray(frames[0])
            out[view] = img
        return out

    def _decode_video_async(self, video_path: str, frame_indices: np.ndarray) -> list[np.ndarray]:
        """Decode video frames using async decoder or video optimizer if available."""
        # Try video optimizer first (provides caching and optimization)
        video_optimizer = self._get_video_optimizer()
        if video_optimizer is not None:
            try:
                return video_optimizer.decode_video_frames(video_path, frame_indices)
            except Exception as e:
                warnings.warn(
                    f"Video optimizer failed: {e}. Falling back to async decoder.",
                    UserWarning,
                    stacklevel=2,
                )
        
        # Fallback to async decoder
        async_decoder = self._get_async_video_decoder()
        if async_decoder is not None:
            request = VideoDecodeRequest(
                video_path=video_path,
                frame_indices=frame_indices,
                video_backend=self._spec.video_backend,
                video_backend_kwargs=self._spec.video_backend_kwargs,
            )
            result = async_decoder.decode_sync(request)
            return result.frames
        else:
            # Final fallback to synchronous decoding
            try:
                from gr00t.utils.video_utils import get_frames_by_indices
                return get_frames_by_indices(
                    video_path,
                    frame_indices,
                    video_backend=self._spec.video_backend,
                    video_backend_kwargs=self._spec.video_backend_kwargs or {},
                )
            except ImportError:
                return []

    def prefetch_video_frames(self, episode_id: int, step: int):
        """Prefetch video frames for future use (non-blocking)."""
        if not self._video_views or self._video_path_pattern is None:
            return
        
        # Prefetch frames for next few steps
        prefetch_steps = [step + i for i in range(1, 4)]  # Prefetch next 3 steps
        
        for prefetch_step in prefetch_steps:
            if prefetch_step >= self._effective_lengths[np.searchsorted(self._cum_lengths, step, side="right")]:
                continue  # Don't prefetch beyond episode end
                
            for view in self._video_views:
                chunk_idx = int(episode_id) // int(self._chunk_size)
                video_filename = self._video_path_pattern.format(
                    episode_chunk=chunk_idx, video_key=view, episode_index=int(episode_id)
                )
                video_path = str(self._dataset_path / video_filename)
                
                # Check if video file exists before prefetching
                if not (self._dataset_path / video_filename).exists():
                    continue
                
                frame_indices = np.asarray([prefetch_step], dtype=np.int64)
                
                # Use video optimizer for prefetching if available
                video_optimizer = self._get_video_optimizer()
                if video_optimizer is not None:
                    video_optimizer.prefetch_frames(video_path, frame_indices)
                else:
                    # Fallback to async decoder prefetch
                    async_decoder = self._get_async_video_decoder()
                    if async_decoder is not None:
                        request = VideoDecodeRequest(
                            video_path=video_path,
                            frame_indices=frame_indices,
                            video_backend=self._spec.video_backend,
                            video_backend_kwargs=self._spec.video_backend_kwargs,
                        )
                        async_decoder.prefetch(request)

    def get_video_decode_stats(self) -> dict[str, Any]:
        """Get video decoding statistics."""
        async_decoder = self._get_async_video_decoder()
        if async_decoder is not None:
            return async_decoder.get_stats()
        return {}

    def prefetch_episodes(self, episode_ids: List[int]):
        """Prefetch episodes into cache for better performance."""
        if self._distributed_cache is not None and self._spec.enable_cache_warming:
            if self._use_gr00t_modality_config:
                # Prefetch using gr00t loader
                self._distributed_cache.prefetch(
                    episode_ids,
                    lambda eid: self._episode_loader[int(eid)]
                )
            else:
                # Prefetch using fallback loader
                def _load_episode(eid: int):
                    chunk_idx = int(eid) // int(self._chunk_size)
                    parquet_filename = self._data_path_pattern.format(episode_chunk=chunk_idx, episode_index=int(eid))
                    parquet_path = self._dataset_path / parquet_filename
                    
                    if not parquet_path.exists():
                        return None
                    
                    cols = ["observation.state", "action", "instruction.text"]
                    try:
                        return pd.read_parquet(parquet_path, columns=cols)
                    except Exception:
                        return pd.read_parquet(parquet_path)
                
                self._distributed_cache.prefetch(episode_ids, _load_episode)

    def get_episode_cache_stats(self) -> dict[str, Any]:
        """Get episode caching statistics."""
        if self._distributed_cache is not None:
            return self._distributed_cache.get_stats()
        return {}

    def get_video_optimization_stats(self) -> dict[str, Any]:
        """Get video optimization statistics."""
        stats = {}
        
        video_optimizer = self._get_video_optimizer()
        if video_optimizer is not None:
            stats["decode_stats"] = video_optimizer.get_stats().__dict__
            stats["cache_info"] = video_optimizer.get_cache_info()
        
        async_decoder = self._get_async_video_decoder()
        if async_decoder is not None:
            stats["async_decoder"] = async_decoder.get_stats()
        
        return stats

    def optimize_video_backend_for_dataset(self) -> bool:
        """
        Optimize video backend configuration based on dataset characteristics.
        
        Returns:
            True if optimization was performed, False otherwise
        """
        if (not self._spec.optimize_video_for_dataset or 
            not self._video_optimizer_config['enabled'] or
            not self._video_views or 
            self._video_path_pattern is None):
            if self._video_path_pattern is None and self._video_views:
                logging.warning("Video path pattern not available, skipping video backend optimization")
            return False
        
        try:
            # Collect sample video paths and frame counts
            sample_videos = []
            sample_frame_counts = []
            
            # Sample first few episodes
            for i, episode_id in enumerate(self._episode_ids[:5]):
                if i >= 3:  # Limit to 3 samples for performance
                    break
                
                # Get video path for first view
                view = self._video_views[0]
                chunk_idx = int(episode_id) // int(self._chunk_size)
                video_filename = self._video_path_pattern.format(
                    episode_chunk=chunk_idx, video_key=view, episode_index=int(episode_id)
                )
                video_path = str(self._dataset_path / video_filename)
                
                if (self._dataset_path / video_filename).exists():
                    sample_videos.append(video_path)
                    # Use effective length as frame count estimate
                    ep_idx = np.where(self._episode_ids == episode_id)[0][0]
                    sample_frame_counts.append(int(self._effective_lengths[ep_idx]))
            
            if sample_videos:
                # Get video optimizer and optimize configuration
                video_optimizer = self._get_video_optimizer()
                if video_optimizer is not None:
                    optimized_config = video_optimizer.optimize_for_dataset(
                        sample_videos, sample_frame_counts
                    )
                    
                    # Update video optimizer with optimized config
                    from openpi.training.video_backend_optimizer import shutdown_global_video_optimizer, get_global_video_optimizer
                    shutdown_global_video_optimizer()
                    self._video_optimizer = get_global_video_optimizer(optimized_config)
                    
                    return True
                
        except Exception as e:
            warnings.warn(
                f"Video backend optimization failed: {e}. Using default configuration.",
                UserWarning,
                stacklevel=2,
            )
        
        return False

    def get_action_processing_stats(self) -> dict[str, Any]:
        """Get action processing performance statistics."""
        if self._parallel_action_processor is not None:
            return self._parallel_action_processor.get_performance_stats()
        return {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, step = self._global_to_episode_step(int(index))

        sample: dict[str, Any] = {}

        if self._use_gr00t_modality_config:
            df = self._get_episode_df(episode_id)
            vla = self._extract_step_data(
                episode_data=df,
                step_index=step,
                modality_configs=self._modality_configs,
                embodiment_tag=self._embodiment_tag_class(self._spec.embodiment_tag),
                allow_padding=False,
                stats_key=self._stats_key,  # Pass stats_key to enable action transforms
            )

            # Images -> LeRobot-style flat keys.
            # Normalize all image keys to observation.images.{view} format
            images = self._select_images(vla.images)
            for view_name, img in images.items():
                # Strip prefix if present, then add it back consistently
                clean_view = view_name.replace("observation.images.", "")
                sample[f"observation.images.{clean_view}"] = img

            # Apply action representation transforms if processor is available
            if self._processor is not None:
                # Use parallel processing if available, otherwise fall back to sequential
                if self._parallel_action_processor is not None:
                    try:
                        # Use parallel processor for better performance
                        processed_actions_list, processed_states_list = self._parallel_action_processor.process_single(
                            vla.actions,
                            vla.states,
                            self._spec.embodiment_tag,
                            self._stats_key,
                        )
                        sample["observation.state"] = self._concat_state(processed_states_list)
                        sample["actions"] = self._concat_actions(processed_actions_list)
                    except Exception as e:
                        warnings.warn(
                            f"Parallel action processing failed: {e}. Falling back to sequential processing.",
                            UserWarning,
                            stacklevel=2,
                        )
                        # Fallback to sequential processing
                        processed_states = self._processor.apply_state(
                            vla.states,
                            embodiment_tag=self._spec.embodiment_tag,
                            stats_key=self._stats_key,
                        )
                        processed_actions = self._processor.apply_action(
                            vla.actions,
                            embodiment_tag=self._spec.embodiment_tag,
                            state=vla.states,
                            stats_key=self._stats_key,
                        )
                        sample["observation.state"] = self._concat_state(processed_states)
                        sample["actions"] = self._concat_actions(processed_actions)
                else:
                    # Sequential processing
                    processed_states = self._processor.apply_state(
                        vla.states,
                        embodiment_tag=self._spec.embodiment_tag,
                        stats_key=self._stats_key,
                    )
                    processed_actions = self._processor.apply_action(
                        vla.actions,
                        embodiment_tag=self._spec.embodiment_tag,
                        state=vla.states,
                        stats_key=self._stats_key,
                    )
                    sample["observation.state"] = self._concat_state(processed_states)
                    sample["actions"] = self._concat_actions(processed_actions)
            else:
                # No processor - use raw state and actions
                sample["observation.state"] = self._concat_state(vla.states)
                sample["actions"] = self._concat_actions(vla.actions)

            # Prompt (optional)
            if vla.text is not None and (self._spec.language_key is None or self._spec.language_key):
                sample["prompt"] = np.asarray(vla.text)
        else:
            # Fallback: read parquet directly and decode videos using async video decoder.
            df = self._get_episode_df_fallback(episode_id)

            # Validate required columns exist
            required_cols = ["observation.state", "action"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Episode {episode_id} missing required columns: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

            # Images (current frame only) - use async decoding
            for view in self._video_views:
                if self._video_path_pattern is None:
                    continue
                chunk_idx = int(episode_id) // int(self._chunk_size)
                video_filename = self._video_path_pattern.format(
                    episode_chunk=chunk_idx, video_key=view, episode_index=int(episode_id)
                )
                video_path = str(self._dataset_path / video_filename)
                
                # Validate video file exists
                if not (self._dataset_path / video_filename).exists():
                    raise FileNotFoundError(
                        f"Video file not found: {self._dataset_path / video_filename}\n"
                        f"Expected pattern: {self._video_path_pattern}\n"
                        f"Episode: {episode_id}, View: {view}, Chunk: {chunk_idx}"
                    )
                
                # Use async video decoder
                frames = self._decode_video_async(
                    video_path, 
                    np.asarray([step], dtype=np.int64)
                )
                
                if len(frames) > 0:
                    # Normalize view name to observation.images.{view} format
                    clean_view = view.replace("observation.images.", "")
                    sample[f"observation.images.{clean_view}"] = np.asarray(frames[0])

            # State
            sample["observation.state"] = np.asarray(df["observation.state"].iloc[step], dtype=np.float32)

            # Actions - validate horizon doesn't exceed episode length
            horizon_indices = [step + d for d in self._action_delta_indices]
            if max(horizon_indices) >= len(df):
                raise IndexError(
                    f"Action horizon index {max(horizon_indices)} exceeds episode length {len(df)} "
                    f"for episode {episode_id}, step {step}. This should have been filtered during "
                    f"dataset initialization."
                )
            action_chunk = np.stack(
                [np.asarray(df["action"].iloc[i], dtype=np.float32) for i in horizon_indices], 
                axis=0
            )
            sample["actions"] = action_chunk

            # Prompt from instruction.text (if present)
            if "instruction.text" in df.columns:
                sample["prompt"] = np.asarray(df["instruction.text"].iloc[step])

            # Prefetch next frames in background (non-blocking)
            self.prefetch_video_frames(episode_id, step)
            
            # Prefetch nearby episodes for cache warming
            if (self._distributed_cache is not None and 
                self._spec.enable_cache_warming and 
                np.random.random() < 0.1):  # 10% chance to trigger prefetch
                # Find nearby episodes to prefetch
                current_ep_idx = np.where(self._episode_ids == episode_id)[0]
                if len(current_ep_idx) > 0:
                    ep_idx = current_ep_idx[0]
                    # Prefetch next 2-3 episodes
                    next_episodes = []
                    for i in range(1, 4):
                        if ep_idx + i < len(self._episode_ids):
                            next_episodes.append(int(self._episode_ids[ep_idx + i]))
                    if next_episodes:
                        self.prefetch_episodes(next_episodes)

        return sample

    def _make_episode_cache(self):
        """Create LRU cache function for episode loading."""
        if self._distributed_cache is not None:
            # Use intelligent distributed cache
            def _cached(episode_id: int):
                return self._distributed_cache.get(
                    episode_id, 
                    lambda eid: self._episode_loader[int(eid)]
                )
            return _cached
        else:
            # Fallback to simple LRU cache
            maxsize = max(1, int(self._spec.episode_cache_size))

            @lru_cache(maxsize=maxsize)
            def _cached(episode_id: int):
                return self._episode_loader[int(episode_id)]

            return _cached

    def _ensure_episode_cache(self):
        """Lazily initialize episode cache per worker process."""
        if not hasattr(self, "_episode_cache_fn"):
            self._episode_cache_fn = self._make_episode_cache()
        return self._episode_cache_fn

    def _get_episode_df(self, episode_id: int):
        """Get episode dataframe using cached loader."""
        return self._ensure_episode_cache()(episode_id)

    def _make_episode_cache_fallback(self):
        """Create LRU cache function for fallback episode loading."""
        if self._distributed_cache is not None:
            # Use intelligent distributed cache
            def _cached(episode_id: int):
                def _load_episode(eid: int):
                    chunk_idx = int(eid) // int(self._chunk_size)
                    parquet_filename = self._data_path_pattern.format(episode_chunk=chunk_idx, episode_index=int(eid))
                    parquet_path = self._dataset_path / parquet_filename
                    
                    # Validate parquet file exists
                    if not parquet_path.exists():
                        raise FileNotFoundError(
                            f"Parquet file not found: {parquet_path}\n"
                            f"Expected pattern: {self._data_path_pattern}\n"
                            f"Episode: {eid}, Chunk: {chunk_idx}"
                        )
                    
                    # Load minimal columns needed for openpi training.
                    cols = ["observation.state", "action", "instruction.text"]
                    # Some datasets may omit instruction.text; pandas will error if column missing.
                    try:
                        return pd.read_parquet(parquet_path, columns=cols)
                    except Exception:
                        return pd.read_parquet(parquet_path)
                
                return self._distributed_cache.get(episode_id, _load_episode)
            
            return _cached
        else:
            # Fallback to simple LRU cache
            maxsize = max(1, int(self._spec.episode_cache_size))

            @lru_cache(maxsize=maxsize)
            def _cached(episode_id: int):
                chunk_idx = int(episode_id) // int(self._chunk_size)
                parquet_filename = self._data_path_pattern.format(episode_chunk=chunk_idx, episode_index=int(episode_id))
                parquet_path = self._dataset_path / parquet_filename
                
                # Validate parquet file exists
                if not parquet_path.exists():
                    raise FileNotFoundError(
                        f"Parquet file not found: {parquet_path}\n"
                        f"Expected pattern: {self._data_path_pattern}\n"
                        f"Episode: {episode_id}, Chunk: {chunk_idx}"
                    )
                
                # Load minimal columns needed for openpi training.
                cols = ["observation.state", "action", "instruction.text"]
                # Some datasets may omit instruction.text; pandas will error if column missing.
                try:
                    return pd.read_parquet(parquet_path, columns=cols)
                except Exception:
                    return pd.read_parquet(parquet_path)

            return _cached

    def _ensure_episode_cache_fallback(self):
        """Lazily initialize fallback episode cache per worker process."""
        if not hasattr(self, "_episode_cache_fallback_fn"):
            self._episode_cache_fallback_fn = self._make_episode_cache_fallback()
        return self._episode_cache_fallback_fn

    def _get_episode_df_fallback(self, episode_id: int):
        """Get episode dataframe using cached fallback loader."""
        return self._ensure_episode_cache_fallback()(episode_id)
