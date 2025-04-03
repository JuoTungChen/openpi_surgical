import numpy as np
import torch
import os
import random
import h5py
import sys
from torch.utils.data import TensorDataset, DataLoader
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
from random import randint
from PIL import Image

mean = np.array([0.0003197277930221558, 1.195600434743172e-05, 1.452023685598307e-05, # delta position gripper 1 
                0.027635409825060946, 0.020087635780962523, 0.009108920863709338, # delta rotation gripper 1
                0.5556995659711689, # absolute jaw angle gripper 1
                -0.00013999599430910913, 2.8947720678494104e-05, 0.0001063209696798343, # delta position gripper 2 
                -0.0008475984176586737, 0.0005478266527301205, -0.003212338329158976, # delta rotation gripper 2
                -0.3222132455187185]) # absolute jaw angle gripper 2
std = np.array([0.01, 0.01, 0.01, 
                0.10578696941952981, 0.07144912379533436, 0.14779914683345105, 
                0.4788636163948364, 
                0.01, 0.01, 0.01, 
                0.018864122464884177, 0.018980285093698757, 0.030213947171999354, 
                0.1479516882199184])

min = np.array([-0.0289371745242912, -0.0192301987888057, -0.0219202609471326,
                 -0.9664272117215005, -0.6826452632171087, -1.2596365853517446, 
                 -0.349096, 
                 -0.019208095197308695, -0.010359199487475799, -0.022508599100696206, 
                 -0.6264236858334745, -0.38042939793142716, -1.013891189892466, 
                 -0.3490660000000002])
                
max = np.array([0.0305917430876664, 0.016248057993992898, 0.027286879786947098,
                1.0051323566433583, 0.7390426745889305, 1.4184644234148385, 
                1.4579498842824292, 
                0.014087676909864701, 0.019160732064643797, 0.01822373460177029, 
                0.5068614814663617, 0.5059128923386381, 0.7254057523100611, 
                1.397807299955499])

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
    
    # List all 'tissue_*' directories in the main folder
    tissue_folders = [
        f for f in os.listdir(main_folder)
        if f.startswith('tissue_') and os.path.isdir(os.path.join(main_folder, f))
    ]
    
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
        robot_base_dir,
        action_horizon = 50,
        cutting_action_pad_size = 10
        ):

        super().__init__()
        self.robot_base_dir = robot_base_dir
        self.action_horizon = action_horizon
        self.cutting_action_pad_size = cutting_action_pad_size
        self.fps = 30
        
        # Get list of episodes
        self.episode_list = get_robot_episodes(self.robot_base_dir)
        
        # Create flattened list of all timesteps across episodes
        self.flattened_indices = []
        self.episode_lengths = []
        
        # Open each episode to get its length
        # Dictionary to store unique instructions and their indices
        self.instruction_to_idx = {}
        curr_idx = 0
        
        for episode_path, instruction, _ in self.episode_list:
            # Add instruction to dict if not seen before
            if instruction not in self.instruction_to_idx:
                self.instruction_to_idx[instruction] = curr_idx
                curr_idx += 1
                
            store = zarr.ZipStore(episode_path, mode='r')
            zarr_store = zarr.group(store=store)
            kinematics = zarr_store['kinematics'][:]
            episode_len = len(kinematics)
            self.episode_lengths.append(episode_len)
            store.close()
            
        # Create mapping from flat index to (episode_idx, timestep)
        curr_offset = 0
        for episode_idx, length in enumerate(self.episode_lengths):
            for ts in range(length):
                self.flattened_indices.append((episode_idx, ts))
            curr_offset += length
        
        # psm = patient side manipulator
        # qpos = current pose read from the robot
        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]

        # sp = setpoint (i.e. when you teleoperate, it generate setpoints for the robot to reach)
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
        
        # open the Zarr store using ZipStore
        store = zarr.ZipStore(episode_path, mode='r')
        zarr_store = zarr.group(store=store)

        kinematics = zarr_store['kinematics'][:]
        df = pd.DataFrame(kinematics)
        episode_len = len(df)
        img_idx = start_ts

        # for cutting tasks, length of the kinematics data extend longer than images, so image index must be capped
        if (instruction == "go to the cutting position left tube" or instruction == "go to the cutting position right tube") and start_ts >= episode_len - self.cutting_action_pad_size:
            img_idx = episode_len - self.cutting_action_pad_size - 1

        # get images
        img_l = np.array(zarr_store['left'][img_idx]) / 255.0    # da vinci endoscope image
        img_lw = np.array(zarr_store['endo_psm2'][img_idx]) / 255.0  # left wrist camera view image from PSM2
        img_rw = np.array(zarr_store['endo_psm1'][img_idx]) / 255.0 # right wrist camera view

        ## rectify rotation of the right wrist cam (for some trials, it was incorrectly placed)
        rotate_ids = [5, 6, 8, 12, 13, 14, 18]
        if tissue_id in rotate_ids:
            angle = -52.0
            img_rw = rotate_image(img_rw, angle)
            shift_x, shift_y = 10, 0 
            img_rw = shift_image(img_rw, shift_x, shift_y)

        # read actions and qpos
        qpos_psm1 = df[self.header_name_qpos_psm1].iloc[start_ts].to_numpy()
        action_psm1 = df[self.header_name_actions_psm1].iloc[start_ts:start_ts + self.action_horizon].to_numpy()
        qpos_psm2 = df[self.header_name_qpos_psm2].iloc[start_ts].to_numpy()
        action_psm2 = df[self.header_name_actions_psm2].iloc[start_ts:start_ts + self.action_horizon].to_numpy()

        diff_psm1 = None
        diff_psm2 = None
        # compute hybrid-relative actions. see: https://surgical-robot-transformer.github.io/
        diff_psm1 = self.compute_diff_actions(qpos_psm1, action_psm1)
        diff_psm2 = self.compute_diff_actions(qpos_psm2, action_psm2)

        # stack the actions along column dim
        action = np.column_stack((diff_psm1, diff_psm2))

        # normalize data
        action = self.normalize_actions(action)

        # set current poses to zeros (dvrk kinematics unreliable)
        qpos = np.zeros(14)

        dataset_dict = {}
        
        dataset_dict["observation.images.left"] = img_l
        # dataset_dict["observation.images.right"] = img_l
        dataset_dict["observation.images.endo_psm1"] = img_rw
        dataset_dict["observation.images.endo_psm2"] = img_lw

        dataset_dict["observation.state"] = qpos
        dataset_dict["actions"] = action
        dataset_dict["timestamp"] = start_ts / self.fps
        dataset_dict["frame_index"] = start_ts
        dataset_dict["episode_index"] = episode_idx
        dataset_dict["index"] = index
        dataset_dict["task_index"] = self.instruction_to_idx[instruction]
        dataset_dict["prompt"] = instruction
        
        dataset_dict["action_is_pad"] = [False if (start_ts + i) < len(df) else True for i in range(self.action_horizon)]


        store.close()

        # Return:
        # img_l: Left endoscopic image (H x W x C)
        # img_lw: Left wrist camera image (H x W x C)
        # img_rw: Right wrist camera image (H x W x C)
        # action: Normalized actions (action_horizon x 14)
        # qpos: Zeroed current positions (14,)
        # instruction: Language instruction for the episode

        # (['observation.state', 'action', 'observation.images.left', 'observation.images.right',
        #  'observation.images.endo_psm1', 'observation.images.endo_psm2', 
        # 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'action_is_pad'])
        
        return img_l, img_lw, img_rw, action, qpos, instruction
        

    def compute_diff_actions(self, qpos, action):
        """
        Computes the relative actions with respect to the current position using axis-angle rotation.

        Parameters:
        - qpos: Current pose (array of shape [8] - xyz, xyzw, jaw angle)
        - action: Actions commanded by the user (array of shape [n_actions x 8] - xyz, xyzw, jaw angle)

        Returns:
        - diff_expand: Relative actions with delta translation and delta rotation in axis-angle format.
                    Shape: (n_actions, 7) - [delta_translation, delta_rotation, jaw_angle]
        """
        # Compute the delta translation w.r.t da vinci endoscope tip frame (approx the camera frame)
        delta_translation = action[:, 0:3] - qpos[0:3]  # Shape: (n_actions, 3)

        # Extract quaternions from qpos and action
        quat_init = qpos[3:7]          # Shape: (4,)
        quat_actions = action[:, 3:7]  # Shape: (n_actions, 4)

        # Convert quaternions to Rotation objects
        r_init = R.from_quat(quat_init)
        r_actions = R.from_quat(quat_actions)

        # Compute the relative rotations
        diff_rs = r_init.inv() * r_actions  # Shape: (n_actions,)

        # Convert the rotation differences to rotation vectors (axis-angle representation)
        delta_rotation = diff_rs.as_rotvec()  # Shape: (n_actions, 3)

        # Extract the jaw angle from the action (note: jaw angle is not relative)
        jaw_angle = action[:, -1]  # Shape: (n_actions,)

        # Prepare the final diff array
        delta_action = np.zeros((action.shape[0], 7))  # Shape: (n_actions, 7)

        # Populate the diff_expand array
        delta_action[:, 0:3] = delta_translation       # Delta translation
        delta_action[:, 3:6] = delta_rotation          # Delta rotation (axis-angle)
        delta_action[:, 6] = jaw_angle                 # Jaw angle (not relative)

        return delta_action

    def normalize_actions(self, diffs):
        """
        diffs: n_actions x 14 (delta position [3], delta orientation (axis-angle) [6], jaw angle (absolute) [1]) for both grippers
        return: normalized n_actions x 14 (zero mean unit variance)
        Note: only position and orientation are normalized, jaw angle is kept as is (absolute)
        the min / max value for each param is at the top of this script
        """

        diff_orig = diffs.copy()
        normalized = (diffs - mean) / std
        # replace w/ originals for jaw angles
        normalized[:, 6] = diff_orig[:, 6]
        normalized[:, 13] = diffs[:, 13]
        return normalized
    

"""
Test the EpisodicDatasetDvrkGeneric class.
"""
if __name__ == "__main__":
    # Specify the data directory as needed
    data_dir = "base_chole_clipping_cutting/processed_data_zipped_pi/"
    
    # Create an instance of the dataset
    dataset = EpisodicDatasetDvrkGeneric(data_dir, action_horizon=50)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Visualize a sample from the DataLoader
    for data in dataloader:
        # Unpack data from DataLoader
        # action is: delta position, delta rotation, jaw angle
        img_l, img_lw, img_rw, action, qpos, instruction = data
        
        # Plot images side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_l.squeeze())
        axes[0].set_title("Left Image")
        axes[0].axis("off")
        
        axes[1].imshow(img_lw.squeeze())
        axes[1].set_title("Endoscopic Image PSM2")
        axes[1].axis("off")
        
        axes[2].imshow(img_rw.squeeze())
        axes[2].set_title("Endoscopic Image PSM1")
        axes[2].axis("off")
        
        plt.tight_layout()
    
        # Save the figure to a file
        fig_path = os.path.join("sample.jpg")
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")
        
        # Close the plot to free memory
        plt.close(fig)
        
        # Print out the numerical data for reference
        print("Action:\n", action.shape)
        print("Qpos:\n", qpos.shape)
        print("Instruction:\n", instruction)
        
        # Only show one batch for visualization, then break out of loop
        break