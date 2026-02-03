#!/usr/bin/env python3
"""
Create a custom dummy configuration for testing with minimal memory usage.
This creates a new config that combines the gr00t_local data setup with dummy model variants.
"""

import sys
sys.path.insert(0, '/app/src')

from openpi.training.config import *
import openpi.models.pi0 as pi0

# Create a new minimal memory config
dummy_gr00t_config = TrainConfig(
    name="dummy_gr00t_local",
    model=pi0.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,  # Adjust based on your robot's action space
        action_horizon=4,
        max_token_len=128,  # Reduced from default
    ),
    data=Gr00tLocalLeRobotDataConfig(
        repo_id="local/gr00t_lerobot_dummy",
        dataset_path="/data/open_h_suturing",
        embodiment_tag="dvrk",
        modality_config_path="/home/jchen396/gr00t_n1.6/examples/dVRK/dVRK_config.py",
        language_key=None,
        video_views=[
            "endoscope_left",
            "wrist_left", 
            "wrist_right",
        ],
        base_config=DataConfig(
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "image": {
                                "base_0_rgb": "observation.images.endoscope_left",
                                "left_wrist_0_rgb": "observation.images.wrist_left",
                                "right_wrist_0_rgb": "observation.images.wrist_right",
                            },
                            "state": "observation.state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
    ),
    batch_size=1,
    num_train_steps=1000,  # Reduced for testing
    num_workers=1,
    overwrite=True,
    wandb_enabled=False,  # Disable wandb for testing
)

# Add the config to the global CONFIGS list
CONFIGS.append(dummy_gr00t_config)

print("Created dummy_gr00t_local configuration with minimal memory usage:")
print(f"- Model: Pi0.5 with dummy variants")
print(f"- Paligemma variant: {dummy_gr00t_config.model.paligemma_variant}")
print(f"- Action expert variant: {dummy_gr00t_config.model.action_expert_variant}")
print(f"- Batch size: {dummy_gr00t_config.batch_size}")
print(f"- Max token length: {dummy_gr00t_config.model.max_token_len}")
print(f"- Action dimension: {dummy_gr00t_config.model.action_dim}")
print(f"- Action horizon: {dummy_gr00t_config.model.action_horizon}")
print("\nThis configuration should use <1GB of GPU memory.")