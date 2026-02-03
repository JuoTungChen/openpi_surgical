#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from openpi.training.gr00t_lerobot_dataset import Gr00tDatasetSpec, Gr00tLeRobotTorchDataset

print(f"Dataset module location: {Gr00tLeRobotTorchDataset.__module__}")
print(f"Dataset file location: {Gr00tLeRobotTorchDataset.__init__.__code__.co_filename}")

# Create the same spec as used in training
spec = Gr00tDatasetSpec(
    dataset_path="/home/iulian/chole_ws/data/open_h_suturing",
    embodiment_tag="dvrk",
    modality_config_path="/home/iulian/chole_ws/src/gr00t_n1.6/examples/dVRK/dVRK_config.py",
    action_horizon=50,  # This should match the config
)

print(f"Creating dataset with spec:")
print(f"  dataset_path: {spec.dataset_path}")
print(f"  embodiment_tag: {spec.embodiment_tag}")
print(f"  action_horizon: {spec.action_horizon}")
print(f"  modality_config_path: {spec.modality_config_path}")

try:
    dataset = Gr00tLeRobotTorchDataset(spec)
    print(f"Dataset created successfully!")
    print(f"Dataset length: {len(dataset)}")
    
    if hasattr(dataset, '_use_gr00t_modality_config'):
        print(f"Use gr00t modality config: {dataset._use_gr00t_modality_config}")
    
    if hasattr(dataset, '_episode_loader'):
        print(f"Episode loader length: {len(dataset._episode_loader)}")
        print(f"Episode loader type: {type(dataset._episode_loader)}")
    
    if hasattr(dataset, '_episode_ids'):
        print(f"Episode IDs: {dataset._episode_ids}")
        print(f"Episode IDs length: {len(dataset._episode_ids)}")
    if hasattr(dataset, '_episode_lengths'):
        print(f"Episode lengths: {dataset._episode_lengths}")
    if hasattr(dataset, '_effective_episode_lengths'):
        print(f"Effective lengths: {dataset._effective_episode_lengths}")
    if hasattr(dataset, '_total_steps'):
        print(f"Total steps: {dataset._total_steps}")
        
    # Check action horizon impact
    print(f"Action horizon: {spec.action_horizon}")
    
except Exception as e:
    print(f"Error creating dataset: {e}")
    import traceback
    traceback.print_exc()