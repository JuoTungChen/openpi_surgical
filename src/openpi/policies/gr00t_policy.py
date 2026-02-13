import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import config as _config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Gr00tPolicyConfig:
    """Configuration for GR00T-aware OpenPI inference.

    This wraps an OpenPI model with GR00T's StateActionProcessor so inference
    mirrors training behavior when GR00T action transforms are enabled.
    """

    # Training config name (e.g., "pi05_gr00t_local").
    train_config_name: str
    # Checkpoint directory containing OpenPI params.
    checkpoint_dir: str
    # GR00T modality config file used during training (registers embodiment).
    modality_config_path: str
    # Embodiment tag string (e.g., "dvrk").
    embodiment_tag: str
    # Consolidated, keyed stats JSON saved during training.
    stats_path: str
    # Stats key to use (repo_id = dataset folder name).
    stats_key: str
    # Optional default prompt to inject if none is provided.
    default_prompt: str | None = None
    # Optional kwargs for model sampling (temperature, top-k, etc.).
    sample_kwargs: dict[str, Any] | None = None


class Gr00tPolicy(_base_policy.BasePolicy):
    """OpenPI policy wrapper that applies GR00T state/action transforms.

    This policy expects **raw** observations:
      - `image`: dict of image arrays keyed by OpenPI image keys
      - `state`: raw concatenated state vector (dataset format)
      - `prompt` (optional): language prompt

    It performs:
      1) GR00T state normalization (percentile stats + modality config)
      2) OpenPI model inference (predicts normalized action chunk)
      3) GR00T action unnormalization + hybrid-relative → absolute conversion

    Output:
      - `actions`: raw action chunk (horizon, dim) in dataset format
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        model_config: _model.BaseModelConfig,
        modality_configs: dict[str, Any],
        embodiment_tag: str,
        stats: dict[str, dict[str, Any]],
        stats_key: str,
        default_prompt: str | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from gr00t.data.state_action.state_action_processor import StateActionProcessor

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._rng = jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}

        self._embodiment_tag = embodiment_tag
        self._stats_key = stats_key
        self._modality_configs = modality_configs

        # Initialize GR00T processor with the same modality config and stats.
        self._processor = StateActionProcessor(
            modality_configs={embodiment_tag: modality_configs[embodiment_tag]},
            statistics=stats,
            use_percentiles=True,
            clip_outliers=True,
            apply_sincos_state_encoding=False,
            use_relative_action=True,
        )
        self._processor.eval()

        # Cache ordering + dims for deterministic split/concat.
        modality = modality_configs[embodiment_tag]
        self._state_keys = list(modality["state"].modality_keys)
        action_cfg = modality["action"]
        pass_through = set(action_cfg.pass_through_keys or [])
        self._action_keys = [k for k in action_cfg.modality_keys if k not in pass_through]
        self._action_horizon = len(action_cfg.delta_indices)

        stats_entry = stats[stats_key]
        self._state_dims = {key: len(stats_entry["state"][key]["mean"]) for key in self._state_keys}
        action_dim_from_stats = {key: len(stats_entry["action"][key]["mean"][0]) for key in self._action_keys}
        action_dims: dict[str, int] = {}
        if action_cfg.action_configs is not None:
            from gr00t.data.types import (
                ActionFormat,
                EEF_XYZ_ROT6D_DIM,
                ROT6D_DIM,
                XYZ_DIM,
            )

            format_dims = {
                ActionFormat.XYZ: XYZ_DIM,
                ActionFormat.ROT6D: ROT6D_DIM,
                ActionFormat.XYZ_ROT6D: EEF_XYZ_ROT6D_DIM,
                ActionFormat.XYZ_ROTVEC: XYZ_DIM + 3,
            }
            for key, cfg in zip(action_cfg.modality_keys, action_cfg.action_configs, strict=True):
                if key in pass_through:
                    continue
                if cfg.format == ActionFormat.DEFAULT:
                    action_dims[key] = action_dim_from_stats[key]
                else:
                    action_dims[key] = format_dims[cfg.format]
        else:
            action_dims = action_dim_from_stats
        self._action_dims = action_dims

        # Determine which state index represents "current" timestep for concatenation.
        state_deltas = list(modality["state"].delta_indices)
        self._state_zero_index = state_deltas.index(0) if 0 in state_deltas else len(state_deltas) - 1

        # Reuse model transforms (tokenization, padding, resizing).
        model_transforms = _config.ModelTransformFactory(default_prompt)(model_config)
        self._input_transform = _transforms.compose(
            (
                _transforms.EnsureImageMask(),
                *model_transforms.inputs,
            )
        )

    def _split_state(self, state_vec: np.ndarray) -> dict[str, np.ndarray]:
        """Split a concatenated state vector into per-key arrays."""
        cursor = 0
        state_dict: dict[str, np.ndarray] = {}
        for key in self._state_keys:
            dim = self._state_dims[key]
            state_dict[key] = np.asarray(state_vec[cursor : cursor + dim], dtype=np.float32)
            cursor += dim
        return state_dict

    def _concat_state(self, state_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate per-key state arrays into a single vector."""
        parts = []
        for key in self._state_keys:
            arr = np.asarray(state_dict[key], dtype=np.float32)
            if arr.ndim == 1:
                parts.append(arr)
            else:
                parts.append(arr[self._state_zero_index])
        return np.concatenate(parts, axis=-1) if parts else np.zeros((0,), dtype=np.float32)

    def _split_actions(self, action_chunk: np.ndarray) -> dict[str, np.ndarray]:
        """Split a concatenated action chunk into per-key arrays."""
        cursor = 0
        actions: dict[str, np.ndarray] = {}
        for key in self._action_keys:
            dim = self._action_dims[key]
            actions[key] = np.asarray(action_chunk[:, cursor : cursor + dim], dtype=np.float32)
            cursor += dim
        return actions

    def _concat_actions(self, action_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate per-key action arrays into a single action chunk."""
        parts = [np.asarray(action_dict[key], dtype=np.float32) for key in self._action_keys]
        if not parts:
            return np.zeros((self._action_horizon, 0), dtype=np.float32)
        return np.concatenate(parts, axis=-1)

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        """Infer actions with GR00T normalization and unnormalization.

        Args:
            obs: Dictionary with keys "image", "state", optional "prompt".

        Returns:
            Dict with raw `actions` chunk (horizon, dim) in dataset format.
        """
        # Preserve raw state for action unnormalization.
        raw_state_vec = np.asarray(obs["state"], dtype=np.float32)
        raw_state_dict = self._split_state(raw_state_vec)
        for key, value in raw_state_dict.items():
            if value.ndim == 1:
                raw_state_dict[key] = value[None, :]

        # Normalize state with GR00T processor, then re-concatenate.
        normalized_state = self._processor.apply_state(
            raw_state_dict,
            embodiment_tag=self._embodiment_tag,
            stats_key=self._stats_key,
        )
        obs = {**obs, "state": self._concat_state(normalized_state)}

        # Apply OpenPI model transforms (tokenization, resize, padding).
        obs = self._input_transform(obs)

        # Batch and convert to JAX arrays.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs)
        self._rng, sample_rng = jax.random.split(self._rng)
        normalized_actions = self._sample_actions(
            sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs
        )

        # Unbatch and convert to numpy.
        normalized_actions = np.asarray(normalized_actions[0, ...], dtype=np.float32)

        # Split, unnormalize, and convert to raw action space.
        normalized_action_dict = self._split_actions(normalized_actions)
        raw_action_dict = self._processor.unapply_action(
            normalized_action_dict,
            embodiment_tag=self._embodiment_tag,
            state=raw_state_dict,
            stats_key=self._stats_key,
        )
        raw_actions = self._concat_actions(raw_action_dict)

        return {"actions": raw_actions}


def create_gr00t_trained_policy(config: Gr00tPolicyConfig) -> Gr00tPolicy:
    """Create a GR00T-aware OpenPI policy from a checkpoint.

    This helper loads:
      - OpenPI model parameters
      - GR00T modality config
      - Consolidated, keyed stats JSON
    and returns a policy that applies GR00T transforms at inference time.
    """
    train_config = _config.get_config(config.train_config_name)
    checkpoint_dir = Path(config.checkpoint_dir)

    # Load model parameters from checkpoint.
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    # Load and register GR00T modality config (registers into MODALITY_CONFIGS).
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    config_path = Path(config.modality_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Modality config not found: {config_path}")
    if config_path.suffix != ".py":
        raise ValueError(f"Modality config must be a .py file: {config_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("groot_modality_config", config_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to import modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if config.embodiment_tag not in MODALITY_CONFIGS:
        raise KeyError(
            f"Embodiment '{config.embodiment_tag}' not found in GR00T modality configs. "
            f"Available: {list(MODALITY_CONFIGS.keys())}"
        )

    # Load consolidated stats (keyed by repo_id).
    stats_path = Path(config.stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    stats = json.loads(stats_path.read_text())
    if config.stats_key not in stats:
        raise KeyError(f"Stats key '{config.stats_key}' not found in {stats_path}. Available: {list(stats.keys())}")

    return Gr00tPolicy(
        model,
        model_config=train_config.model,
        modality_configs=MODALITY_CONFIGS,
        embodiment_tag=config.embodiment_tag,
        stats=stats,
        stats_key=config.stats_key,
        default_prompt=config.default_prompt,
        sample_kwargs=config.sample_kwargs,
    )
