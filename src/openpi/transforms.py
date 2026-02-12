from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class Gr00tUnnormalize(DataTransformFn):
    """GR00T-style denormalization using StateActionProcessor from GR00T repo.
    
    This transform directly uses GR00T's StateActionProcessor.unapply_action()
    to denormalize actions, ensuring 100% consistency with GR00T's inference pipeline.
    
    Required parameters:
    - modality_config_path: Path to GR00T modality config (e.g., dVRK_config.py)
    - percentile_stats_path: Path to percentile_stats.json from GR00T training
    - embodiment_tag: Embodiment identifier (e.g., "dvrk", "gr1")
    
    Optional parameters:
    - stats_key: Override for statistics lookup (defaults to embodiment_tag)
    - use_percentiles: Whether to use percentile normalization (default: True)
    - clip_outliers: Whether to clip normalized values (default: True)
    - use_relative_action: Whether actions are in relative format (default: True)
    
    Example usage:
        transform = Gr00tUnnormalize(
            modality_config_path="/path/to/dVRK_config.py",
            percentile_stats_path="/path/to/percentile_stats.json",
            embodiment_tag="dvrk",
        )
    """
    modality_config_path: str
    percentile_stats_path: str
    embodiment_tag: str
    stats_key: str | None = None
    use_percentiles: bool = True
    clip_outliers: bool = True
    use_relative_action: bool = True
    
    # Lazy-loaded processor (set in __post_init__)
    _processor: object | None = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize GR00T StateActionProcessor with modality config and stats."""
        try:
            # Import necessary modules
            import sys
            import os
            import json
            import importlib
            from pathlib import Path
            
            # Import GR00T's StateActionProcessor
            from gr00t.data.state_action.state_action_processor import StateActionProcessor
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
            
            # Load modality config by importing the Python file
            # This registers the config in MODALITY_CONFIGS as a side effect
            config_path = Path(self.modality_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Modality config not found: {config_path}")
            
            # Add parent directory to sys.path temporarily
            parent_dir = str(config_path.parent.resolve())
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Import the module (this registers the config)
            module_name = config_path.stem
            importlib.import_module(module_name)
            
            # Get the registered config
            if self.embodiment_tag not in MODALITY_CONFIGS:
                raise ValueError(
                    f"Embodiment tag '{self.embodiment_tag}' not found in MODALITY_CONFIGS. "
                    f"Available: {list(MODALITY_CONFIGS.keys())}"
                )
            
            modality_configs = {self.embodiment_tag: MODALITY_CONFIGS[self.embodiment_tag]}
            
            # Load statistics
            with open(self.percentile_stats_path, "r") as f:
                percentile_stats = json.load(f)
            
            # Add embodiment_tag metadata if not present (for per-dataset stats)
            stats_key = self.stats_key or self.embodiment_tag
            if stats_key not in percentile_stats:
                # Assume stats are at root level - wrap them
                statistics = {stats_key: percentile_stats}
            else:
                statistics = percentile_stats
            
            # Initialize processor
            object.__setattr__(self, "_processor", StateActionProcessor(
                modality_configs=modality_configs,
                statistics=statistics,
                use_percentiles=self.use_percentiles,
                clip_outliers=self.clip_outliers,
                apply_sincos_state_encoding=False,
                use_relative_action=self.use_relative_action,
            ))
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GR00T StateActionProcessor: {e}\n"
                f"Make sure GR00T is available at the expected path and configs are valid."
            ) from e

    def __call__(self, data: DataDict) -> DataDict:
        """Denormalize actions using GR00T's StateActionProcessor."""
        if self._processor is None:
            raise RuntimeError("StateActionProcessor not initialized")
        
        # Extract actions from data
        if "actions" not in data:
            return data
        
        actions = data["actions"]
        
        # Convert OpenPI action format to GR00T format
        # OpenPI: (T, D) numpy array where D is concatenated dimensions of all groups
        # GR00T: dict[str, np.ndarray] with separate arrays for each action group
        
        # Get modality keys and their dimensions from processor
        modality_keys = self._processor.modality_configs[self.embodiment_tag]["action"].modality_keys
        stats_key = self.stats_key or self.embodiment_tag
        norm_params = self._processor.norm_params[stats_key]["action"]
        
        # Get modality config to check action formats
        from gr00t.data.types import ActionType, ActionFormat
        action_configs = self._processor.modality_configs[self.embodiment_tag]["action"].action_configs
        
        # Debug: print actual action shape
        print(f"[Gr00tUnnormalize] Input action shape: {actions.shape}")
        print(f"[Gr00tUnnormalize] Modality keys: {modality_keys}")
        
        # Calculate expected dimension from modality config
        expected_total_dim = 0
        dim_info = []
        for idx, key in enumerate(modality_keys):
            params = norm_params[key]
            action_config = action_configs[idx] if action_configs and idx < len(action_configs) else None
            is_eef_xyz_rot6d = (
                action_config is not None
                and action_config.type == ActionType.EEF
                and action_config.format == ActionFormat.XYZ_ROT6D
            )
            
            if is_eef_xyz_rot6d:
                dim = 9  # 3 (xyz) + 6 (rot6d)
            else:
                if "q02" in params:
                    q02 = params["q02"]
                    dim = 1 if isinstance(q02, (int, float)) else (q02.shape[0] if q02.ndim == 1 else q02.shape[-1])
                elif "min" in params:
                    min_val = params["min"]
                    dim = 1 if isinstance(min_val, (int, float)) else (min_val.shape[0] if min_val.ndim == 1 else min_val.shape[-1])
                else:
                    raise ValueError(f"Cannot determine dimension for action group '{key}'")
            
            dim_info.append(f"{key}={dim}")
            expected_total_dim += dim
        
        print(f"[Gr00tUnnormalize] Expected dimensions: {', '.join(dim_info)} (total={expected_total_dim})")
        
        # Handle dimension mismatch: if model outputs more dims than needed, truncate
        if actions.shape[-1] > expected_total_dim:
            print(f"[Gr00tUnnormalize] WARNING: Model outputs {actions.shape[-1]} dims but only {expected_total_dim} needed. Using first {expected_total_dim} dims.")
            actions = actions[..., :expected_total_dim]
        elif actions.shape[-1] < expected_total_dim:
            raise ValueError(
                f"Model outputs {actions.shape[-1]} dims but {expected_total_dim} dims required for modality config"
            )
        
        # Split actions into dict format based on dimensions from stats
        action_dict = {}
        current_idx = 0
        
        for idx, key in enumerate(modality_keys):
            # Get dimension for this group from normalization params
            params = norm_params[key]
            
            # Determine actual action dimension based on action config
            action_config = action_configs[idx] if action_configs and idx < len(action_configs) else None
            is_eef_xyz_rot6d = (
                action_config is not None
                and action_config.type == ActionType.EEF
                and action_config.format == ActionFormat.XYZ_ROT6D
            )
            
            if is_eef_xyz_rot6d:
                # For XYZ_ROT6D: stats only cover xyz (3D), but action is xyz+rot6d (9D)
                dim = 9  # 3 (xyz) + 6 (rot6d)
            else:
                # For other types, dimension matches stats
                if "q02" in params:
                    q02 = params["q02"]
                    if isinstance(q02, (int, float)):
                        dim = 1
                    elif hasattr(q02, 'shape'):
                        dim = q02.shape[0] if q02.ndim == 1 else q02.shape[-1]
                    else:
                        dim = len(q02)
                elif "min" in params:
                    min_val = params["min"]
                    if isinstance(min_val, (int, float)):
                        dim = 1
                    elif hasattr(min_val, 'shape'):
                        dim = min_val.shape[0] if min_val.ndim == 1 else min_val.shape[-1]
                    else:
                        dim = len(min_val)
                else:
                    raise ValueError(f"Cannot determine dimension for action group '{key}'")
            
            # Extract this group's actions
            action_dict[key] = actions[..., current_idx:current_idx + dim]
            current_idx += dim
        
        # Prepare state dict for hybrid-relative conversion
        # For hybrid-relative actions, we need current state to convert back to absolute
        state_dict = None
        if "state" in data and data["state"] is not None:
            state = data["state"]
            print(f"[Gr00tUnnormalize] Raw state shape: {state.shape}, dtype: {state.dtype}")
            
            # Split state into dict format matching action groups
            # State should have same structure as actions (concatenated joint values)
            state_dict = {}
            state_idx = 0
            
            # Get state dimensions from modality config
            state_modality_keys = self._processor.modality_configs[self.embodiment_tag]["state"].modality_keys
            state_norm_params = self._processor.norm_params[stats_key]["state"]
            
            print(f"[Gr00tUnnormalize] State modality keys: {state_modality_keys}")
            
            for key in state_modality_keys:
                params = state_norm_params[key]
                if "q02" in params:
                    q02 = params["q02"]
                    dim = 1 if isinstance(q02, (int, float)) else (q02.shape[0] if q02.ndim == 1 else q02.shape[-1])
                elif "min" in params:
                    min_val = params["min"]
                    dim = 1 if isinstance(min_val, (int, float)) else (min_val.shape[0] if min_val.ndim == 1 else min_val.shape[-1])
                else:
                    # Skip if no normalization params
                    continue
                
                # Extract state for this group
                # State is typically (T, D) or just (D,)
                # For hybrid-relative conversion, GR00T expects state with shape (1, D) for single reference
                if state.ndim == 1:
                    state_slice = state[state_idx:state_idx + dim]
                    # Add batch dimension for GR00T's hybrid-relative conversion
                    state_dict[key] = state_slice[np.newaxis, :]  # (D,) -> (1, D)
                else:
                    state_dict[key] = state[..., state_idx:state_idx + dim]
                
                print(f"[Gr00tUnnormalize] State {key}: shape={state_dict[key].shape}, range=[{state_dict[key].min():.4f}, {state_dict[key].max():.4f}]")
                state_idx += dim
        
        # Apply denormalization
        denormalized_dict = self._processor.unapply_action(
            action_dict,
            embodiment_tag=self.embodiment_tag,
            state=state_dict,  # Pass state for hybrid-relative conversion
            stats_key=stats_key,
        )
        
        # Debug: print denormalized action shapes
        for key, value in denormalized_dict.items():
            print(f"[Gr00tUnnormalize] Denormalized {key}: shape={value.shape}, range=[{value.min():.4f}, {value.max():.4f}]")
        
        # Convert back to OpenPI format (concatenate all groups)
        denormalized_actions = np.concatenate([denormalized_dict[k] for k in modality_keys], axis=-1)
        print(f"[Gr00tUnnormalize] Final output shape: {denormalized_actions.shape}")
        print(f"[Gr00tUnnormalize] Output format: absolute xyz + absolute quat for each pose, absolute value for grippers")
        data["actions"] = denormalized_actions
        
        return data


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class EnsureImageMask(DataTransformFn):
    """Ensure `image_mask` exists for all images.

    Some dataset backends provide images but no validity masks. `openpi.models.model.Observation.from_dict`
    expects `image_mask` to be present, so we default to "all valid".
    """

    def __call__(self, data: DataDict) -> DataDict:
        if "image" not in data:
            return data
        if "image_mask" in data:
            return data

        # Derive batch shape from state if present; otherwise use scalar True (unbatched).
        if (state := data.get("state")) is not None and hasattr(state, "shape"):
            batch_shape = state.shape[:-1]
            default_mask = np.ones(batch_shape, dtype=np.bool_)
        else:
            default_mask = np.asarray(True)

        data["image_mask"] = {k: default_mask for k in data["image"].keys()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
