#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')
import pickle
import tempfile

from openpi.training.gr00t_lerobot_dataset import Gr00tDatasetSpec, Gr00tLeRobotTorchDataset

# Create a minimal spec to test pickling
spec = Gr00tDatasetSpec(
    dataset_path="/home/iulian/chole_ws/data/open_h_suturing",
    embodiment_tag="dvrk",
    modality_config_path="/home/iulian/chole_ws/src/gr00t_n1.6/examples/dVRK/dVRK_config.py",
    action_horizon=4,
    video_views=[],
    enable_async_video_decoding=True,
    enable_video_backend_optimization=False,
    optimize_video_for_dataset=False,
    enable_intelligent_caching=False,
    enable_parallel_action_processing=True,
)

print("Creating dataset...")
try:
    dataset = Gr00tLeRobotTorchDataset(spec)
    print(f"Dataset created successfully! Length: {len(dataset)}")
    
    print("Testing pickle...")
    with tempfile.NamedTemporaryFile() as f:
        pickle.dump(dataset, f)
        print("Dataset pickled successfully!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()