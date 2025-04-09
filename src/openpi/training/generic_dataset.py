import numpy as np
import torch
import os
import random
import h5py
import sys
from torch.utils.data import TensorDataset, DataLoader, Sampler
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torchvision import transforms, utils
from tqdm import tqdm
import json
import time
import copy
import re
import zarr
from typing import Union, Optional, Tuple
import numcodecs
import numpy as np
import simplejpeg
import functools
import matplotlib.pyplot as plt
import math
from random import randint
from PIL import Image
from collections import defaultdict, Counter


def _assert_shape(arr: np.ndarray, expected_shape: Tuple[Optional[int], ...]):
    """Asserts that the shape of an array matches the expected shape."""
    assert len(arr.shape) == len(expected_shape), f"Expected shape of length {len(expected_shape)}, but got {len(arr.shape)}"
    for dim, expected_dim in zip(arr.shape, expected_shape):
        if expected_dim is not None:
            assert dim == expected_dim, f"Expected dimension {expected_dim}, but got {dim}"

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def shift_image(image, shift_x, shift_y):
    """Shift the image by the given x and y offsets."""
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image


class JpegCodec(numcodecs.abc.Codec):
    """Codec for JPEG compression.
    Encodes image chunks as JPEGs. Assumes that chunks are uint8 with shape (1, H, W, 3).
    """
    codec_id = "pi_jpeg"

    def __init__(self, quality: int = 95):
        super().__init__()
        self.quality = quality

    def encode(self, buf):
        _assert_shape(buf, (1, None, None, 3))
        assert buf.dtype == "uint8"
        return simplejpeg.encode_jpeg(buf[0], quality=self.quality)

    def decode(self, buf, out=None):
        img = simplejpeg.decode_jpeg(buf, buffer=out)
        return img[np.newaxis, ...]

@functools.cache
def register_codecs():
    """Register the custom codecs."""
    numcodecs.register_codec(JpegCodec)

register_codecs()

def get_robot_episodes(main_folder):
    """
    Get a list of episodes from the robot dataset. 
    It returns a list of tuples, where each tuple contains: the episode path, the language instruction, and the tissue index.
    """
    episode_list = []
    print(f"[DEBUG] Looking for episodes in: {main_folder}")
    
    # List all 'tissue_*' directories in the main folder
    tissue_folders = [
        f for f in os.listdir(main_folder)
        if f.startswith('tissue_') and os.path.isdir(os.path.join(main_folder, f))
    ]
    print("getting robot episodes")
    for tissue_folder in tissue_folders:
        tissue_path = os.path.join(main_folder, tissue_folder)
        
        # Extract tissue index from the tissue folder name
        tissue_index_match = re.search(r'tissue_(\d+)', tissue_folder)
        if tissue_index_match:
            tissue_index = int(tissue_index_match.group(1))
        else:
            tissue_index = None  # or handle as needed
        
        # List all instruction folders within the tissue folder
        instruction_folders = [
            f for f in os.listdir(tissue_path)
            if os.path.isdir(os.path.join(tissue_path, f))
        ]
        
        for instr_folder in instruction_folders:
            instr_path = os.path.join(tissue_path, instr_folder)
            
            # Extract language instruction from the folder name
            # Remove leading numbers and underscores
            instr_name = re.sub(r'^\d+_*', '', instr_folder)
            # Remove digits elsewhere in the string
            instr_name = re.sub(r'\d+', '', instr_name)
            # Remove the word 'recovery' if it exists
            instr_name = instr_name.replace('recovery', '')
            # Replace underscores with spaces
            instr_name = instr_name.replace('_', ' ')
            # Strip leading and trailing whitespace
            instr_name = instr_name.strip()
            
            # List all episode files (zipped files) within the instruction folder
            episode_files = [
                f for f in os.listdir(instr_path)
                if os.path.isfile(os.path.join(instr_path, f)) and f.endswith('.zip')
            ]
            
            for episode_file in episode_files:
                episode_path = os.path.join(instr_path, episode_file)
                episode_list.append((episode_path, instr_name, tissue_index))
    print("robot episodes obtained:", len(episode_list))
    return episode_list

# Example usage: for testing get_robot_episodes()
# Replace '/path/to/your/dataset' with the actual path to your main folder
# episodes = get_robot_episodes("base_chole_clipping_cutting/processed_data_zipped_pi/")

# # Print the list of episodes, their corresponding language instructions, and tissue indices
# for episode_path, instruction, tissue_index in episodes:
#     print(f"Episode Path: {episode_path}")
#     print(f"Language Instruction: {instruction}")
#     print(f"Tissue Index: {tissue_index}\n")

# assert(False)

class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        robot_base_dir_list,
        action_horizon=50,
        cutting_action_pad_size=10
    ):
        super().__init__()

        if isinstance(robot_base_dir_list, str):
            self.robot_base_dir_list = [robot_base_dir_list]
        else:
            self.robot_base_dir_list = robot_base_dir_list

        self.action_horizon = action_horizon
        self.cutting_action_pad_size = cutting_action_pad_size
        self.fps = 30
        self.episode_list = []
        self.flattened_indices = []
        self.episode_lengths = []
        self.instruction_to_idx = {}
        curr_idx = 0

        # Get list of episodes from all directories
        for robot_base_dir in self.robot_base_dir_list:
            self.episode_list.extend(get_robot_episodes(robot_base_dir))

        # Metadata paths for all directories
        meta_paths = [os.path.join(robot_base_dir, "meta.json") for robot_base_dir in self.robot_base_dir_list]
        task_meta_paths = [os.path.join(robot_base_dir, "task_meta.json") for robot_base_dir in self.robot_base_dir_list]

        # Dictionary to store task meta
        task_meta = {}

        # Check if metadata exists for all directories
        if all(os.path.exists(meta_path) and os.path.exists(task_meta_path) for meta_path, task_meta_path in zip(meta_paths, task_meta_paths)):
            print("Metadata files found. Loading from all directories.")
            for meta_path, task_meta_path in zip(meta_paths, task_meta_paths):
                with open(meta_path, "r") as meta_file:
                    meta_data = [json.loads(line) for line in meta_file]
                with open(task_meta_path, "r") as task_meta_file:
                    task_data = [json.loads(line) for line in task_meta_file]

                # Populate episode_lengths and instruction_to_idx from loaded metadata
                self.episode_lengths.extend([entry["length"] for entry in meta_data])
                for entry in task_data:
                    instruction = entry["tasks"][0]
                    if instruction not in self.instruction_to_idx:
                        self.instruction_to_idx[instruction] = curr_idx
                        curr_idx += 1
                        # Initialize task meta entry if not already
                        if instruction not in task_meta:
                            task_meta[instruction] = {
                                "episode_indices": [],
                                "tissue_ids": []
                            }
                    # Append episode index and tissue ID for the task
                    episode_idx = entry["episode_index"]
                    tissue_id = entry.get("tissue_id", None)  # Add tissue_id if it exists
                    task_meta[instruction]["episode_indices"].append(episode_idx)
                    task_meta[instruction]["tissue_ids"].append(tissue_id)
        else:
            print("Metadata files not found for some directories. Generating metadata...")
            for robot_base_dir in self.robot_base_dir_list:
                meta_data = []
                task_data = []
                for episode_idx, (episode_path, instruction, _) in enumerate(tqdm(self.episode_list)):
                    # Ensure the episode belongs to one of the robot_base_dirs
                    if not any(episode_path.startswith(robot_base_dir) for robot_base_dir in self.robot_base_dir_list):
                        continue
                    # Ensure the episode belongs to the current robot_base_dir
                    if not episode_path.startswith(robot_base_dir):
                        continue

                    store = zarr.ZipStore(episode_path, mode='r')
                    zarr_store = zarr.group(store=store)
                    kinematics = zarr_store['kinematics'][:]
                    episode_len = len(kinematics)
                    self.episode_lengths.append(episode_len)
                    store.close()

                    # Append metadata for the current episode
                    meta_data.append({
                        "episode_index": episode_idx,
                        "tasks": [instruction],
                        "length": episode_len
                    })

                    # Add instruction to dict if not seen before
                    if instruction not in self.instruction_to_idx:
                        self.instruction_to_idx[instruction] = curr_idx
                        curr_idx += 1
                        task_data.append({
                            "episode_index": episode_idx,
                            "tasks": [instruction],
                            "tissue_id": instruction,  # You can set this to any relevant tissue ID
                        })

                    # Append the task's episode index and tissue ID to the task_meta
                    if instruction not in task_meta:
                        task_meta[instruction] = {
                            "episode_indices": [],
                            "tissue_ids": []
                        }
                    task_meta[instruction]["episode_indices"].append(episode_idx)
                    task_meta[instruction]["tissue_ids"].append(instruction)  # This should be the actual tissue ID

                # Save metadata to JSON files for the current directory
                meta_path = os.path.join(robot_base_dir, "meta.json")
                task_meta_path = os.path.join(robot_base_dir, "task_meta.json")
                with open(meta_path, "w") as meta_file:
                    for entry in meta_data:
                        meta_file.write(json.dumps(entry) + "\n")
                print(f"Metadata saved to {meta_path}")
                with open(task_meta_path, "w") as task_meta_file:
                    for entry in task_data:
                        task_meta_file.write(json.dumps(entry) + "\n")
                print(f"Task Metadata saved to {task_meta_path}")

        # Save task_meta to a global file for later sampling
        task_meta_global_path = os.path.join(self.robot_base_dir_list[0], "task_meta_global.json")
        with open(task_meta_global_path, "w") as task_meta_global_file:
            json.dump(task_meta, task_meta_global_file)
        print(f"Global Task Metadata saved to {task_meta_global_path}")

        # Create mapping from flat index to (episode_idx, timestep)
        # curr_offset = 0
        # for episode_idx, length in enumerate(self.episode_lengths):
        #     for ts in range(length):
        #         self.flattened_indices.append((episode_idx, ts))
        #     curr_offset += length


        # Sort episode list by (task name) to group tasks together 
        self.episode_list.sort(key=lambda tup: tup[1])

        # Create flattened index + per-task sample indices
        self.flattened_indices = []
        self.task_to_indices = defaultdict(list)

        for episode_idx, (episode_path, instruction, _) in enumerate(self.episode_list):
            episode_len = self.episode_lengths[episode_idx]
            task_idx = self.instruction_to_idx[instruction]

            for ts in range(episode_len):
                flat_idx = len(self.flattened_indices)
                self.flattened_indices.append((episode_idx, ts))
                self.task_to_indices[task_idx].append(flat_idx)

            
        # psm = patient side manipulator
        # qpos = current pose read from the robot
        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                      "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                      "psm1_jaw"]

        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                      "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                      "psm2_jaw"]

        # sp = setpoint (i.e. when you teleoperate, it generates setpoints for the robot to reach)
        self.header_name_actions_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                         "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                         "psm1_jaw_sp"]

        self.header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                         "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                         "psm2_jaw_sp"]

        self.quat_cp_psm1 = ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"]
        self.quat_cp_psm2 = ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]

    def __len__(self):
        return len(self.flattened_indices)

    def __getitem__(self, index):
        # Get episode index and timestep from flattened index
        episode_idx, start_ts = self.flattened_indices[index]
        episode_path, instruction, tissue_id = self.episode_list[episode_idx]

        # Open the Zarr store using ZipStore
        store = zarr.ZipStore(episode_path, mode='r')
        zarr_store = zarr.group(store=store)

        kinematics = zarr_store['kinematics'][:]
        df = pd.DataFrame(kinematics)
        episode_len = len(df)

        # Make sure start_ts does not exceed episode length
        if start_ts >= episode_len:
            start_ts = episode_len - 1

        # For cutting tasks, clip image index
        if instruction in ["go to the cutting position left tube", "go to the cutting position right tube"]:
            max_img_idx = episode_len - self.cutting_action_pad_size - 1
            img_idx = min(start_ts, max_img_idx)
        else:
            img_idx = min(start_ts, episode_len - 1)

        # Get images
        img_l = np.array(zarr_store['left'][img_idx]) / 255.0
        img_r = np.array(zarr_store['right'][img_idx]) / 255.0
        img_lw = np.array(zarr_store['endo_psm2'][img_idx]) / 255.0
        img_rw = np.array(zarr_store['endo_psm1'][img_idx]) / 255.0

        # Fix right wrist cam orientation for certain tissue IDs
        rotate_ids = [5, 6, 8, 12, 13, 14, 18]
        if tissue_id in rotate_ids:
            angle = -52.0
            img_rw = rotate_image(img_rw, angle)
            img_rw = shift_image(img_rw, shift_x=10, shift_y=0)

        # Read actions and qpos
        qpos_psm1 = df[self.header_name_qpos_psm1].iloc[start_ts].to_numpy()
        action_psm1 = df[self.header_name_actions_psm1].iloc[start_ts:start_ts + self.action_horizon].to_numpy()
        qpos_psm2 = df[self.header_name_qpos_psm2].iloc[start_ts].to_numpy()
        action_psm2 = df[self.header_name_actions_psm2].iloc[start_ts:start_ts + self.action_horizon].to_numpy()

        qpos = np.concatenate([qpos_psm1, qpos_psm2])
        action = np.column_stack((action_psm1, action_psm2))

        # Pad actions if shorter than action horizon
        if action.shape[0] < self.action_horizon:
            action = np.concatenate((action, np.tile(action[-1], (self.action_horizon - action.shape[0], 1))), axis=0)

        dataset_dict = {
            "observation.images.left": img_l,
            "observation.images.right": img_r,
            "observation.images.endo_psm1": img_rw,
            "observation.images.endo_psm2": img_lw,
            "observation.state": qpos,
            "action": action,
            "timestamp": img_idx / self.fps,
            "frame_index": img_idx,
            "episode_index": episode_idx,
            "index": index,
            "task_index": self.instruction_to_idx[instruction],
            "prompt": instruction,
            "action_is_pad": [False if (img_idx + i) < episode_len else True for i in range(self.action_horizon)],
        }

        store.close()
        return dataset_dict

    

class StratifiedBatchSampler(Sampler):
    def __init__(self, task_to_indices, batch_size, samples_per_task=1, seed=0):
        self.task_to_indices = task_to_indices
        self.task_ids = list(task_to_indices.keys())
        #self.samples_per_task = samples_per_task
        self.batch_size = batch_size
        self.seed = seed
        self.samples_per_task = min(samples_per_task, min(len(v) for v in self.task_to_indices.values()))


    def __iter__(self):
        random.seed(self.seed)

        task_pools = {
            task: random.sample(indices, len(indices))
            for task, indices in self.task_to_indices.items()
        }

        batches = []
        while True:
            batch = []
            for task in self.task_ids:
                pool = task_pools[task]
                if len(pool) < self.samples_per_task:
                    break
                batch.extend([pool.pop() for _ in range(self.samples_per_task)])

            if len(batch) < self.batch_size:
                break

            batches.append(batch)

        return iter(batches)

    def __len__(self):
        min_size = min(len(indices) for indices in self.task_to_indices.values())
        return min_size // self.samples_per_task


"""
Test the EpisodicDatasetDvrkGeneric class.
"""
if __name__ == "__main__":
    # Specify the data directory as needed
    #data_dir = "base_chole_clipping_cutting/processed_data_zipped_pi/"
    data_dir = "../chole_ws/data/base_chole_clipping_cutting/processed_data_zipped_pi/"
    
    # Create an instance of the dataset
    dataset = EpisodicDatasetDvrkGeneric(data_dir, action_horizon=50)

    #create sampler
    samples_per_task = 10
    num_tasks = len(dataset.instruction_to_idx)

    sampler = StratifiedBatchSampler(
        task_to_indices=dataset.task_to_indices,
        batch_size=num_tasks * samples_per_task,
        samples_per_task=samples_per_task,
        seed=42
    )

    # Create a DataLoader
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    

    seen = set()
    # Grab one batch
    for batch_indices in dataloader.batch_sampler:
        # Collect samples from dataset
        batch = [dataset[i] for i in batch_indices]

        overlap = seen.intersection(batch_indices)
        assert len(overlap) == 0, f"Duplicate samples across batches: {overlap}"

        seen.update(batch_indices)

        # Extract task indices
        task_indices = [sample["task_index"] for sample in batch]
        prompts = [sample["prompt"] for sample in batch]
        episode_paths = [dataset.episode_list[sample["episode_index"]][0] for sample in batch]

        #check if unique episodes are sampled
        #assert len(set(episode_paths)) == len(episode_paths), "Duplicate episodes found in the batch!"


        # Show distribution
        task_distribution = Counter(task_indices)
        print("Task distribution in batch", task_distribution)

