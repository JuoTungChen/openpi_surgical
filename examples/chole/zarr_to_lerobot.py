#!/usr/bin/env python
"""
A script to convert robotics data from a single Zarr store into the LeRobot format (v2.1).

This script is designed to process a single Zarr store that contains an entire
dataset, with episode boundaries defined by an `episode_ends` array. It extracts
observations, actions, and state information for each episode and packages them
into a LeRobotDataset, which can then be optionally pushed to the Hugging Face Hub.

Expected Zarr Store Structure:
------------------------------
The script assumes a single Zarr store (e.g., `my_dataset.zarr`) with the following
internal hierarchy. All top-level arrays are expected to be flat, containing
data for all episodes concatenated together. `N` is the total number of steps
across all episodes, and `E` is the total number of episodes.

/
├── action                (Array, shape: (N, 6)): The actions for all steps.
├── observations/
│   └── rgb               (Array, shape: (N, 2, 224, 224, 3)): RGB images for all steps.
├── abs_joint_pos         (Array, shape: (N, 7)): The absolute joint positions for all steps.
├── timestep              (Array, shape: (N,)): The timestamp for each data point.
└── episode_ends          (Array, shape: (E,)): Indices marking the end of each episode.

Usage:
------
To run the script, you can use the following command, pointing to your Zarr store:

    python convert_zarr_to_lerobot.py --data-path /path/to/your/dataset.zarr

To convert and then upload to the Hugging Face Hub:

    python convert_zarr_to_lerobot.py --data-path /path/to/your/dataset.zarr --push-to-hub

Dependencies:
-------------
- lerobot
- tyro
- zarr
- tqdm
"""

import shutil
from pathlib import Path

import zarr
import tqdm
import tyro
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME


def convert_data_to_lerobot(data_path: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts robotics data from a Zarr store into LeRobot dataset format.

    This function processes a single Zarr store containing concatenated episode data
    and transforms it into a structured LeRobotDataset. The conversion handles video
    observations, robot states, and actions while preserving episode boundaries.

    The function creates a new LeRobotDataset with predefined features for robotic
    manipulation tasks, including dual camera views (main and wrist cameras),
    7-DOF joint states, and 6-DOF Cartesian actions.

    Args:
        data_path (Path): Absolute or relative path to the Zarr store directory
            containing the robotics dataset. Must include 'episode_ends' array
            for proper episode segmentation.
        repo_id (str): Unique identifier for the dataset repository on Hugging Face Hub.
            Format should be 'username/dataset-name' for proper organization.
        push_to_hub (bool, optional): Flag to automatically upload the converted
            dataset to Hugging Face Hub after successful conversion. Defaults to False.

    Raises:
        FileNotFoundError: If the specified Zarr store path does not exist.
        KeyError: If required arrays ('episode_ends', 'observations/rgb', etc.)
            are missing from the Zarr store.
        ValueError: If episode boundaries are inconsistent or data shapes don't match
            expected dimensions.

    Note:
        - Existing datasets at the target location will be automatically removed
        - Each episode is processed sequentially to manage memory usage
        - Failed episodes are skipped with error logging to ensure partial success
    """
    # Prepare output directory - remove existing dataset to ensure clean conversion
    final_output_path = LEROBOT_HOME / repo_id
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    # Initialize LeRobotDataset with schema tailored for robotic manipulation tasks
    # Features include dual RGB cameras, 7-DOF joint states, and 6-DOF Cartesian actions
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        video=True,
        robot_type="panda",
        fps=30,
        features={
            "image": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
        },
    )

    # Load and validate the source Zarr store
    print(f"Opening Zarr store at {data_path}")
    try:
        root_zarr = zarr.open(store=str(data_path), mode="r")
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        return

    # Verify required episode boundary information exists
    if "episode_ends" not in root_zarr:
        print(f"Error: `episode_ends` array not found in {data_path}. Cannot determine episode boundaries.")
        return

    # Extract episode metadata for processing loop
    episode_ends = root_zarr["episode_ends"][:]
    num_episodes = len(episode_ends)
    print(f"Found {num_episodes} episodes to convert.")

    # Define consistent task description for all episodes in this dataset
    task_description = "Conduct a liver ultrasound scan"

    # Convert episodes sequentially using episode boundary indices
    start_idx = 0
    for episode_idx in tqdm.tqdm(range(num_episodes), desc="Converting Episodes"):
        try:
            end_idx = episode_ends[episode_idx]

            # Process each timestep within the current episode
            for step_idx in range(start_idx, end_idx):
                # Extract multi-modal observation and action data for this timestep
                frame_data = {
                    "image": root_zarr["observations/rgb"][step_idx][0],  # Main camera view
                    "wrist_image": root_zarr["observations/rgb"][step_idx][1],  # Wrist-mounted camera
                    "state": root_zarr["abs_joint_pos"][step_idx],  # 7-DOF joint positions
                    "action": root_zarr["action"][step_idx],  # 6-DOF Cartesian action
                }
                timestamp = root_zarr["timestep"][step_idx]
                dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

            # Finalize the current episode and persist to storage
            dataset.save_episode()

            # Advance to the next episode's starting position
            start_idx = end_idx

        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            # Clear partial episode data to maintain dataset integrity
            dataset.clear_episode_buffer()

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_path: Path = Path("path/to/your/dataset.zarr"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Command-line interface for converting Zarr robotics datasets to LeRobot format.

    This function serves as the primary entry point for the conversion script,
    providing validation, user feedback, and orchestrating the conversion process.
    It handles command-line argument parsing through tyro and provides informative
    error messages for common issues.

    The function performs preliminary validation before delegating to the core
    conversion logic, ensuring a smooth user experience with clear feedback
    about the conversion process and any potential issues.

    Args:
        data_path (Path, optional): File system path to the source Zarr store
            containing robotics episode data. Must be a valid directory or file
            with .zarr extension. Defaults to "path/to/your/dataset.zarr".
        repo_id (str, optional): Target repository identifier for Hugging Face Hub
            in the format 'username/dataset-name'. Used for both local storage
            organization and remote repository naming. Defaults to placeholder value.
        push_to_hub (bool, optional): Whether to automatically upload the converted
            dataset to Hugging Face Hub upon successful conversion. Requires valid
            HF credentials and network access. Defaults to False.

    Returns:
        None: Function performs side effects (file operations, network uploads)
            and provides status updates through console output.

    Examples:
        Basic conversion:
            python zarr_to_lerobot.py --data-path ./my_dataset.zarr --repo-id user/dataset

        Convert and upload:
            python zarr_to_lerobot.py --data-path ./data.zarr --repo-id user/data --push-to-hub

    Note:
        - Validates input paths before processing to provide early error feedback
        - Warns users about default placeholder values to prevent accidental usage
        - All console output is designed to be informative for debugging and monitoring
    """
    # Validate input path exists before attempting conversion
    if not data_path.exists():
        print(f"Error: The provided Zarr store does not exist: {data_path}")
        print("Please provide a valid path to your Zarr store.")
        return

    # Warn about placeholder values to prevent accidental usage
    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    # Execute the core conversion process
    convert_data_to_lerobot(data_path, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)
