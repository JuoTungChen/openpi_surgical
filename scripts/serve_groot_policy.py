#!/usr/bin/env python3
"""Serve a GR00T-style trained policy over websocket.

This script serves a policy trained with GR00T configs using GR00T's StateActionProcessor
for denormalization during inference. The policy is served over a websocket server that
clients can connect to for real-time inference.

Example usage:
    # Serve a GR00T-trained policy with specific modality config
    python scripts/serve_groot_policy.py \\
        --config pi05_gr00t_local \\
        --checkpoint-dir /path/to/checkpoint \\
        --modality-config /path/to/dVRK_config.py \\
        --percentile-stats /path/to/percentile_stats.json \\
        --embodiment-tag dvrk \\
        --port 8000

    # Serve with recording enabled
    python scripts/serve_groot_policy.py \\
        --config pi05_gr00t_local \\
        --checkpoint-dir /path/to/checkpoint \\
        --modality-config /path/to/dVRK_config.py \\
        --percentile-stats /path/to/percentile_stats.json \\
        --embodiment-tag dvrk \\
        --record \\
        --port 8000
"""

import dataclasses
import logging
import socket
from pathlib import Path

import json

import tyro

from openpi.policies import policy as _policy
from openpi.policies.gr00t_policy import Gr00tPolicyConfig, create_gr00t_trained_policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    """Arguments for serving a GR00T-style policy."""

    # Training config name (e.g., "pi05_gr00t_local").
    config: str

    # Checkpoint directory containing trained model weights.
    checkpoint_dir: str

    # Path to GR00T modality config file (e.g., "/path/to/dVRK_config.py").
    # This must be the same config used during training.
    modality_config: str

    # Path to percentile_stats.json from GR00T training.
    # This file contains q02/q98 statistics for denormalization.
    percentile_stats: str

    # Embodiment tag matching the modality config (e.g., "dvrk", "gr1").
    embodiment_tag: str

    # Optional stats key override. If None, uses the only key in the stats file.
    stats_key: str | None = None

    # Default prompt to use if not provided in observation data.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Record the policy's behavior for debugging.
    record: bool = False

    # Directory to save policy recordings (if record=True).
    record_dir: str = "policy_records"


def create_groot_policy(args: Args) -> _policy.BasePolicy:
    """Create a policy with GR00T StateActionProcessor denormalization.

    Args:
        args: Command-line arguments specifying policy configuration.

    Returns:
        Policy object ready for serving.

    Raises:
        FileNotFoundError: If checkpoint, modality config, or stats file not found.
        ValueError: If configuration is invalid.
    """
    # Validate paths
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    modality_config_path = Path(args.modality_config)
    if not modality_config_path.exists():
        raise FileNotFoundError(f"Modality config not found: {modality_config_path}")

    percentile_stats_path = Path(args.percentile_stats)
    if not percentile_stats_path.exists():
        raise FileNotFoundError(f"Percentile stats not found: {percentile_stats_path}")

    logging.info("Loading GR00T-style policy...")
    logging.info("  Config: %s", args.config)
    logging.info("  Checkpoint: %s", checkpoint_path)
    logging.info("  Modality config: %s", modality_config_path)
    logging.info("  Percentile stats: %s", percentile_stats_path)
    logging.info("  Embodiment tag: %s", args.embodiment_tag)
    stats_key = args.stats_key
    if stats_key is None:
        stats = json.loads(percentile_stats_path.read_text())
        if len(stats) == 1:
            stats_key = next(iter(stats.keys()))
            logging.info("Resolved stats_key to '%s' from stats file", stats_key)
        else:
            raise ValueError(
                f"stats_key is required when stats file contains multiple entries. Available keys: {list(stats.keys())}"
            )

    policy_cfg = Gr00tPolicyConfig(
        train_config_name=args.config,
        checkpoint_dir=str(checkpoint_path),
        modality_config_path=str(modality_config_path),
        embodiment_tag=args.embodiment_tag,
        stats_path=str(percentile_stats_path),
        stats_key=stats_key,
        default_prompt=args.default_prompt,
    )
    policy = create_gr00t_trained_policy(policy_cfg)

    logging.info("✓ Policy loaded successfully with GR00T StateActionProcessor")
    return policy


def main(args: Args) -> None:
    """Main entry point for serving GR00T-style policy.

    Args:
        args: Command-line arguments.
    """
    # Create policy with GR00T denormalization
    policy = create_groot_policy(args)
    policy_metadata = {
        "policy_type": "gr00t",
        "train_config": args.config,
        "embodiment_tag": args.embodiment_tag,
    }

    # Wrap with recorder if requested
    if args.record:
        logging.info("Recording enabled. Saving to: %s", args.record_dir)
        policy = _policy.PolicyRecorder(policy, args.record_dir)

    # Get network information
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating websocket policy server")
    logging.info("  Host: %s", hostname)
    logging.info("  IP: %s", local_ip)
    logging.info("  Port: %d", args.port)
    logging.info("  Metadata: %s", policy_metadata)

    # Create and start websocket server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )

    logging.info("=" * 80)
    logging.info("Server started! Connect clients to ws://%s:%d", local_ip, args.port)
    logging.info("=" * 80)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error("Server error: %s", e)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
