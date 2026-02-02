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


class Gr00tLeRobotTorchDataset(torch.utils.data.Dataset):
    """
    Random-access dataset backed by gr00t's LeRobotEpisodeLoader + modality configs.

    Each element corresponds to a (episode_id, step_index) pair, where step_index is a
    valid base index for the configured action horizon (i.e., we don't sample beyond
    the end of an episode).
    """

    def __init__(self, spec: Gr00tDatasetSpec):
        _require_gr00t()

        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self._spec = spec
        
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
                
                spec_module = importlib.util.spec_from_file_location("temp_modality_config", config_path)
                if spec_module and spec_module.loader:
                    module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(module)
        
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
                
                # Try to load statistics for action representation transformations
                stats_path = Path(spec.dataset_path) / "meta" / "percentile_stats.json"
                if stats_path.exists():
                    with open(stats_path, 'r') as f:
                        self._statistics = json.load(f)
                    # Use provided stats_key or infer from embodiment_tag
                    self._stats_key = spec.stats_key or spec.embodiment_tag
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
                        # Initialize StateActionProcessor with loaded statistics
                        self._processor = StateActionProcessor(
                            modality_configs={spec.embodiment_tag: self._modality_configs},
                            statistics=self._statistics,
                            use_percentiles=True,  # Use percentile-based normalization
                            clip_outliers=True,
                            apply_sincos_state_encoding=False,
                            use_relative_action=True,  # Enable relative/hybrid-relative conversions
                        )
                        self._processor.eval()  # Set to eval mode (no data augmentation)
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
            effective = np.maximum(raw_lengths - self._max_action_delta, 0)
            self._effective_lengths = effective
            self._cum_lengths = np.cumsum(self._effective_lengths, dtype=np.int64)
            self._total_steps = int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

            if "video" in self._modality_configs:
                self._video_views = spec.video_views or list(self._modality_configs["video"].modality_keys)
            else:
                self._video_views = spec.video_views or []

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

            self._action_delta_indices = list(range(int(spec.action_horizon)))
            self._max_action_delta = int(max(self._action_delta_indices)) if self._action_delta_indices else 0

            effective = np.maximum(raw_lengths - self._max_action_delta, 0)
            self._effective_lengths = effective
            self._cum_lengths = np.cumsum(self._effective_lengths, dtype=np.int64)
            self._total_steps = int(self._cum_lengths[-1]) if len(self._cum_lengths) else 0

            # Views come from spec (required for fallback).
            self._video_views = spec.video_views or []

            # State and action come from raw columns.
            self._state_keys = []
            self._action_keys = []

        if self._total_steps <= 0:
            raise ValueError(
                "No valid (episode, step) pairs found. "
                "Check your dataset path and that episodes are longer than the action horizon."
            )

    def __len__(self) -> int:
        return self._total_steps

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
                # StateActionProcessor requires state dict for hybrid-relative conversion
                # Apply state processing (normalization)
                processed_states = self._processor.apply_state(
                    vla.states,
                    embodiment_tag=self._spec.embodiment_tag,
                    stats_key=self._stats_key,
                )
                
                # Apply action processing (hybrid-relative conversion + normalization)
                processed_actions = self._processor.apply_action(
                    vla.actions,
                    embodiment_tag=self._spec.embodiment_tag,
                    state=vla.states,  # Pass raw states as reference for relative conversion
                    stats_key=self._stats_key,
                )
                
                # Concatenate processed state and actions
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
            # Fallback: read parquet directly and decode videos using gr00t video utils.
            from gr00t.utils.video_utils import get_frames_by_indices
            from pathlib import Path

            df = self._get_episode_df_fallback(episode_id)

            # Validate required columns exist
            required_cols = ["observation.state", "action"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Episode {episode_id} missing required columns: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

            # Images (current frame only)
            for view in self._video_views:
                if self._video_path_pattern is None:
                    continue
                chunk_idx = int(episode_id) // int(self._chunk_size)
                video_filename = self._video_path_pattern.format(
                    episode_chunk=chunk_idx, video_key=view, episode_index=int(episode_id)
                )
                video_path = self._dataset_path / video_filename
                
                # Validate video file exists
                if not video_path.exists():
                    raise FileNotFoundError(
                        f"Video file not found: {video_path}\n"
                        f"Expected pattern: {self._video_path_pattern}\n"
                        f"Episode: {episode_id}, View: {view}, Chunk: {chunk_idx}"
                    )
                
                frames = get_frames_by_indices(
                    str(video_path),
                    np.asarray([step], dtype=np.int64),
                    video_backend=self._spec.video_backend,
                    video_backend_kwargs=self._spec.video_backend_kwargs or {},
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

        return sample

    def _make_episode_cache(self):
        """Create LRU cache function for episode loading."""
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
