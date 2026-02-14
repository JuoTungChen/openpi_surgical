"""OpenPI rollout evaluation using GR00T-style server/client inference.

This script mirrors GR00T's eval flow:
  1) Start a policy server (OpenPI model + GR00T transforms)
  2) Create a dataset-driven client
  3) Roll out full episodes from a LeRobot dataset
  4) Compare predicted action chunks to ground truth in raw action space

It intentionally avoids any validation split logic. Episode IDs must be explicit.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required for rollout_eval.py") from exc

from openpi.models import model as _model
from openpi.policies.gr00t_policy import Gr00tPolicyConfig, create_gr00t_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi_client.websocket_client_policy import WebsocketClientPolicy


@dataclasses.dataclass
class RolloutConfig:
    """YAML configuration for rollout evaluation."""

    # Identification
    eval_name: str

    # Model + stats
    train_config_name: str
    checkpoint_dir: str
    stats_path: str
    stats_key: str  # repo_id (dataset folder name)

    # Dataset + embodiment
    dataset_path: str
    embodiment_tag: str
    modality_config_path: str
    video_views: list[str]
    action_horizon: int

    # Episode selection
    episode_ids: list[int]
    num_episodes: int

    # Inference
    default_prompt: str | None = None
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_timeout_ms: int = 30000

    # Output
    output_dir: str = "./rollout_results"
    save_plots: bool = False
    inference_stride: int | None = None

    # Optional explicit mapping from OpenPI image keys to dataset views
    # Example: {"base_0_rgb": "endoscope_left", "left_wrist_0_rgb": "wrist_left"}
    image_key_map: dict[str, str] | None = None


def load_config(path: str | Path) -> RolloutConfig:
    """Load YAML config into a RolloutConfig dataclass."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RolloutConfig(**data)


def build_image_key_map(video_views: list[str], image_key_map: dict[str, str] | None) -> dict[str, str]:
    """Create a mapping from OpenPI image keys to dataset view names."""
    if image_key_map:
        return image_key_map

    # Default mapping uses OpenPI's IMAGE_KEYS order.
    image_keys = list(_model.IMAGE_KEYS)
    if len(video_views) < len(image_keys):
        raise ValueError(f"Not enough video views for model inputs: {video_views} vs {image_keys}")
    return {key: view for key, view in zip(image_keys, video_views, strict=True)}


def load_modality_config(path: str | Path) -> None:
    """Import a GR00T modality config file to register embodiment configs."""
    import importlib.util

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Modality config not found: {config_path}")
    if config_path.suffix != ".py":
        raise ValueError(f"Modality config must be a .py file: {config_path}")

    spec = importlib.util.spec_from_file_location("groot_modality_config", config_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to import modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def concat_state(
    state_dict: dict[str, np.ndarray],
    state_keys: list[str],
    state_zero_index: int,
) -> np.ndarray:
    """Concatenate per-key state arrays into a single vector."""
    parts: list[np.ndarray] = []
    for key in state_keys:
        arr = np.asarray(state_dict[key], dtype=np.float32)
        parts.append(arr if arr.ndim == 1 else arr[state_zero_index])
    return np.concatenate(parts, axis=-1) if parts else np.zeros((0,), dtype=np.float32)


def concat_actions(action_dict: dict[str, np.ndarray], action_keys: list[str]) -> np.ndarray:
    """Concatenate per-key action arrays into a single (H, D) array."""
    parts = [np.asarray(action_dict[key], dtype=np.float32) for key in action_keys]
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(parts, axis=-1)


def select_images(
    vla_images: dict[str, list[np.ndarray]],
    image_key_map: dict[str, str],
) -> dict[str, np.ndarray]:
    """Select the current frame for each view and map to OpenPI image keys."""
    images: dict[str, np.ndarray] = {}
    for image_key, view_name in image_key_map.items():
        frames = vla_images.get(view_name)
        if frames is None or len(frames) == 0:
            raise KeyError(f"Missing frames for view '{view_name}'")
        images[image_key] = np.asarray(frames[0])
    return images


def compute_chunk_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[float, float, int]:
    """Compute MSE/MAE for the overlapping portion of two action chunks."""
    if predicted.shape[1] != ground_truth.shape[1]:
        raise ValueError(f"Action dim mismatch: predicted {predicted.shape}, gt {ground_truth.shape}")
    horizon = min(predicted.shape[0], ground_truth.shape[0])
    pred = predicted[:horizon]
    gt = ground_truth[:horizon]
    diff = pred - gt
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    count = horizon * pred.shape[1]
    return mse, mae, count


def plot_action_series(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    inference_mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot predicted vs ground-truth action series per dimension."""
    import matplotlib.pyplot as plt

    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Plot input mismatch: predicted {predicted.shape}, ground_truth {ground_truth.shape}")
    if inference_mask.shape[0] != predicted.shape[0]:
        raise ValueError(f"Plot mask mismatch: inference_mask {inference_mask.shape}, predicted {predicted.shape}")

    steps, dims = predicted.shape
    ncols = 4
    nrows = max(1, int(np.ceil(dims / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    x = np.arange(steps)
    for idx in range(nrows * ncols):
        ax = axes_arr[idx // ncols][idx % ncols]
        if idx >= dims:
            ax.axis("off")
            continue
        ax.plot(x, ground_truth[:, idx], label="gt", linewidth=1.2)
        ax.plot(x, predicted[:, idx], label="pred", linewidth=1.0, alpha=0.8)
        if inference_mask.any():
            ax.plot(
                x[inference_mask],
                predicted[inference_mask, idx],
                linestyle="None",
                marker="o",
                markersize=2,
                label="infer",
            )
        ax.set_title(f"dim {idx}")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
    axes_arr[0][0].legend(loc="upper right", fontsize="small")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def start_server(cfg: RolloutConfig) -> WebsocketClientPolicy:
    """Start the policy server in a background thread and return a client."""
    policy_cfg = Gr00tPolicyConfig(
        train_config_name=cfg.train_config_name,
        checkpoint_dir=cfg.checkpoint_dir,
        modality_config_path=cfg.modality_config_path,
        embodiment_tag=cfg.embodiment_tag,
        stats_path=cfg.stats_path,
        stats_key=cfg.stats_key,
        default_prompt=cfg.default_prompt,
    )
    policy = create_gr00t_trained_policy(policy_cfg)

    server = WebsocketPolicyServer(policy=policy, host=cfg.server_host, port=cfg.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Client blocks until server is ready.
    return WebsocketClientPolicy(host=cfg.server_host, port=cfg.server_port)


def run_rollout(cfg: RolloutConfig) -> dict[str, Any]:
    """Run rollout evaluation for a single config file."""
    load_modality_config(cfg.modality_config_path)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    if cfg.embodiment_tag not in MODALITY_CONFIGS:
        raise KeyError(f"Embodiment '{cfg.embodiment_tag}' not found. Available: {list(MODALITY_CONFIGS.keys())}")

    modality_configs = MODALITY_CONFIGS[cfg.embodiment_tag]
    action_cfg = modality_configs["action"]
    action_keys = list(action_cfg.modality_keys)
    max_delta = int(max(action_cfg.delta_indices))
    if cfg.action_horizon != len(action_cfg.delta_indices):
        raise ValueError(
            f"action_horizon mismatch: config={cfg.action_horizon} modality_config={len(action_cfg.delta_indices)}"
        )

    state_deltas = list(modality_configs["state"].delta_indices)
    state_zero_index = state_deltas.index(0) if 0 in state_deltas else len(state_deltas) - 1
    state_keys = list(modality_configs["state"].modality_keys)

    inference_stride = cfg.inference_stride or cfg.action_horizon
    if inference_stride <= 0:
        raise ValueError(f"inference_stride must be positive, got {inference_stride}")

    image_key_map = build_image_key_map(cfg.video_views, cfg.image_key_map)
    client = start_server(cfg)

    loader = LeRobotEpisodeLoader(
        dataset_path=cfg.dataset_path,
        modality_configs=modality_configs,
        video_backend="torchcodec",
        skip_video=False,
    )

    results: dict[str, Any] = {
        "eval_name": cfg.eval_name,
        "dataset_path": cfg.dataset_path,
        "embodiment_tag": cfg.embodiment_tag,
        "stats_key": cfg.stats_key,
        "num_episodes": cfg.num_episodes,
        "per_episode": [],
    }

    if cfg.num_episodes > len(cfg.episode_ids):
        raise ValueError(f"num_episodes={cfg.num_episodes} exceeds episode_ids length {len(cfg.episode_ids)}")
    episode_ids = cfg.episode_ids[: cfg.num_episodes]
    for episode_id in episode_ids:
        df = loader[episode_id]
        usable_length = len(df) - max_delta
        if usable_length <= 0:
            continue

        ep_mse = 0.0
        ep_mae = 0.0
        ep_count = 0
        pred_series: list[np.ndarray] = []
        gt_series: list[np.ndarray] = []
        inference_mask: list[bool] = []

        for step in range(usable_length):
            vla = extract_step_data(
                df,
                step,
                modality_configs,
                EmbodimentTag(cfg.embodiment_tag),
                allow_padding=False,
                stats_key=cfg.stats_key,
            )

            obs = {
                "image": select_images(vla.images, image_key_map),
                "state": concat_state(vla.states, state_keys, state_zero_index),
            }
            if vla.text is not None:
                obs["prompt"] = vla.text
            gt_actions = concat_actions(vla.actions, action_keys)
            run_inference = step % inference_stride == 0
            inference_mask.append(run_inference)

            if run_inference:
                response = client.infer(obs)
                pred_actions = np.asarray(response["actions"], dtype=np.float32)
                mse, mae, count = compute_chunk_metrics(pred_actions, gt_actions)
                ep_mse += mse * count
                ep_mae += mae * count
                ep_count += count
                if cfg.save_plots and pred_actions.size:
                    pred_series.append(np.asarray(pred_actions[0], dtype=np.float32))
            elif cfg.save_plots and gt_actions.size:
                pred_series.append(np.full(gt_actions.shape[1], np.nan, dtype=np.float32))

            if cfg.save_plots and gt_actions.size:
                gt_series.append(np.asarray(gt_actions[0], dtype=np.float32))

        if ep_count == 0:
            continue

        if cfg.save_plots and pred_series and gt_series:
            pred_arr = np.stack(pred_series, axis=0)
            gt_arr = np.stack(gt_series, axis=0)
            mask_arr = np.asarray(inference_mask, dtype=bool)
            plot_path = Path(cfg.output_dir) / "plots" / f"episode_{int(episode_id)}_actions.png"
            plot_action_series(
                pred_arr,
                gt_arr,
                mask_arr,
                plot_path,
                title=f"Episode {int(episode_id)} actions (t+0)",
            )

        results["per_episode"].append(
            {
                "episode_id": int(episode_id),
                "episode_length": int(len(df)),
                "mse": ep_mse / ep_count,
                "mae": ep_mae / ep_count,
            }
        )

    # Aggregate metrics.
    if results["per_episode"]:
        results["avg_mse"] = float(np.mean([e["mse"] for e in results["per_episode"]]))
        results["avg_mae"] = float(np.mean([e["mae"] for e in results["per_episode"]]))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rollout_results.json"
    output_path.write_text(json.dumps(results, indent=2))

    return results


def main(config_path: str) -> None:
    """Entry point for running a rollout config."""
    cfg = load_config(config_path)
    start = time.time()
    run_rollout(cfg)
    elapsed = time.time() - start
    print(f"[rollout_eval] completed in {elapsed:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python eval_scripts/rollout_eval.py <config.yaml>")
    main(sys.argv[1])
