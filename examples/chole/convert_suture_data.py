"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro


import dataclasses
import torch

import os
import numpy as np
import pandas as pd
import zarr
import numcodecs
import simplejpeg
from PIL import Image
import functools
from typing import List, Literal
import time

# from multiprocessing import Pool, cpu_count

# DATSET_NAME = "chole_data_lerobot_1"  # Name of the output dataset, also used for the Hugging Face Hub
# CHOLE_DATA_HOME = "/cis/home/sschmi46/chole_ws/data/base_chole_clipping_cutting"  # Name of the output dataset, also used for the Hugging Face Hub


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 15
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _assert_shape(arr: np.ndarray, expected_shape: tuple[int | None, ...]):
    """Asserts that the shape of an array matches the expected shape."""
    assert len(arr.shape) == len(expected_shape), (arr.shape, expected_shape)
    for dim, expected_dim in zip(arr.shape, expected_shape):
        if expected_dim is not None:
            assert dim == expected_dim, (arr.shape, expected_shape)

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

def read_images(image_dir: str, file_pattern: str) -> np.ndarray:
    """Reads images from a directory into a NumPy array."""
    images = []
    ## count images in the dir
    num_images = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
    for idx in range(num_images):
        filename = os.path.join(image_dir, file_pattern.format(idx))
        if not os.path.exists(filename):
            print(f"Warning: {filename} does not exist.")
            continue
        img = Image.open(filename)
        img_array = np.array(img)[..., :3]  # Ensure 3 channels
        images.append(img_array)
    if images:
        return np.stack(images)
    else:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

def process_episode(dataset, episode_path, output_base_dir, states_name, actions_name):
    """Processes a single episode, save the data to lerobot format"""
    # episode_path, output_base_dir = args  # Unpack arguments

    # Paths to image directories
    left_dir = os.path.join(episode_path, 'left_img_dir')
    right_dir = os.path.join(episode_path, 'right_img_dir')
    psm1_dir = os.path.join(episode_path, 'endo_psm1')
    psm2_dir = os.path.join(episode_path, 'endo_psm2')
    csv_file = os.path.join(episode_path, 'ee_csv.csv')

    # Read CSV to determine the number of frames (excluding header)
    df = pd.read_csv(csv_file)

    # Read images from each camera
    left_images = read_images(left_dir, 'frame{:06d}_left.jpg')
    right_images = read_images(right_dir, 'frame{:06d}_right.jpg')
    psm1_images = read_images(psm1_dir, 'frame{:06d}_psm1.jpg')
    psm2_images = read_images(psm2_dir, 'frame{:06d}_psm2.jpg')
    print(left_images.shape, right_images.shape, psm1_images.shape, psm2_images.shape)
    num_frames = min(len(df), left_images.shape[0])

    # Read kinematics data and convert to structured array with headers
    kinematics_data = np.array(
        [tuple(row) for row in df.to_numpy()],
        dtype=[(col, df[col].dtype.str) for col in df.columns]
    )
    print(episode_path)
    # print(kinematics_data.dtype.names)
    # print(kinematics_data["psm1_pose.position.x"])
    # print(kinematics_data["psm1_pose.position.x"][0])

    for i in range(num_frames):
        frame = {
            "observation.state": np.hstack([
                kinematics_data[n][i] for n in states_name
            ]),
            "action": np.hstack([
                kinematics_data[n][i] for n in actions_name
            ]),
        }

        for cam_name, images in [('left', left_images), ('right', right_images), ('endo_psm1', psm1_images), ('endo_psm2', psm2_images)]:
            if images.size > 0:
                frame[f"observation.images.{cam_name}"] = images[i]

        dataset.add_frame(frame)

    return dataset

    # # Create Zarr store
    # relative_path = os.path.relpath(episode_path)
    # zarr_path = os.path.join(output_base_dir, relative_path + '.zarr')
    # os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    # zarr_store = zarr.open_group(zarr_path, mode='w')

    # # Set up compressor
    # compressor = JpegCodec(quality=90)

    # # Store images
    # for cam_name, images in [('left', left_images), ('right', right_images),
    #                          ('endo_psm1', psm1_images), ('endo_psm2', psm2_images)]:
    #     if images.size > 0:
    #         image_store = zarr_store.create_dataset(
    #             cam_name,
    #             shape=images.shape,
    #             chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
    #             dtype='uint8',
    #             compressor=compressor
    #         )
    #         image_store[:] = images

    # # Store kinematics data as structured array
    # zarr_store.create_dataset(
    #     'kinematics',
    #     data=kinematics_data,
    #     dtype=kinematics_data.dtype
    # )


def process_all_episodes(base_dir: str, tissue_indices: List[int], output_base_dir: str):
    """Processes all episodes for given tissue indices using multiprocessing."""

    idx = 0



    if (LEROBOT_HOME / output_base_dir).exists():
       print("removing existing dataset")
       shutil.rmtree(LEROBOT_HOME / output_base_dir)
    
    episode_paths = []
    states_name = [
        "psm1_pose.position.x",
        "psm1_pose.position.y",
        "psm1_pose.position.z",
        "psm1_pose.orientation.x",
        "psm1_pose.orientation.y",
        "psm1_pose.orientation.z",
        "psm1_pose.orientation.w",
        "psm2_pose.position.x",
        "psm2_pose.position.y",
        "psm2_pose.position.z",
        "psm2_pose.orientation.x",
        "psm2_pose.orientation.y",
        "psm2_pose.orientation.z",
        "psm2_pose.orientation.w",
    ]
    actions_name = [
        "psm1_sp.position.x",
        "psm1_sp.position.y",
        "psm1_sp.position.z",
        "psm1_sp.orientation.x",
        "psm1_sp.orientation.y",
        "psm1_sp.orientation.z",
        "psm1_sp.orientation.w",
        "psm2_sp.position.x",
        "psm2_sp.position.y",
        "psm2_sp.position.z",
        "psm2_sp.orientation.x",
        "psm2_sp.orientation.y",
        "psm2_sp.orientation.z",
        "psm2_sp.orientation.w",
    ]
    # create empty dataset
    dataset = create_empty_dataset(
        repo_id=output_base_dir,
        robot_type="dvrk",
        states_name=states_name,
        actions_name=actions_name,
        mode="image",
        dataset_config=DEFAULT_DATASET_CONFIG,
    )
    # input("dataset created successful, press Enter to continue...")

    # measure time taken to complete the process
    start_time = time.time()

    # for tissue_idx in tissue_indices:
    tissue_dir = os.path.join(base_dir, f'tissue_{tissue_indices[idx]}')
    if not os.path.exists(tissue_dir):
        print(f"Warning: {tissue_dir} does not exist.")
        exit()
        # continue

    for subtask_name in os.listdir(tissue_dir):
        subtask_dir = os.path.join(tissue_dir, subtask_name)
        if not os.path.isdir(subtask_dir):
            continue

        for episode_name in os.listdir(subtask_dir):
            episode_dir = os.path.join(subtask_dir, episode_name)
            if not os.path.isdir(episode_dir):
                continue
            dataset = process_episode(dataset, episode_dir, output_base_dir, states_name, actions_name)
            # input("episode processed successful, press Enter to continue...")

            # Collect the episode path and output directory
            # episode_paths.append((episode_dir, output_base_dir))

            dataset.save_episode(task=subtask_name)

        # input("subtask processed sucessful, press Enter to continue...")
        print(f"subtask {subtask_name} processed successful, time taken: {time.time() - start_time}")
    print(f"tissue {tissue_indices[idx]} processed successful, time taken: {time.time() - start_time}")
    dataset.consolidate()

    # Determine the number of processes to use
    # num_processes = min(cpu_count() - 15, len(episode_paths))
    # print(f"Processing {len(episode_paths)} episodes with {num_processes} processes.")

    # # Use multiprocessing Pool to process episodes in parallel
    # with Pool(processes=num_processes) as pool:
    #     pool.map(process_episode, episode_paths)




def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    states_name: List[str],
    actions_name: List[str],
    mode: Literal["video", "image"] = "image",
    *,
    # has_velocity: bool = False,
    # has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    cameras = [
        "left",
        "right",
        "endo_psm1",
        "endo_psm2",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(states_name),),
            "names": [
                states_name,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(actions_name),),
            "names": [
                actions_name,
            ],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3) if cam.startswith("endo") else (540, 960, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        }

    # if Path(LEROBOT_HOME / repo_id).exists():
    #     shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        # use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )



if __name__ == "__main__":
    # tyro.cli(main)
    # base_dir = '.'  # Current directory
    # output_base_dir = './processed_data'  # New folder for Zarr files
    base_dir = "/home/iulian/chole_ws/data/Jesse"  # Name of the output dataset, also used for the Hugging Face Hub
    output_base_dir = "/home/iulian/chole_ws/data/suturing_lerobot"  # Name of the output dataset, also used for the Hugging Face Hub

    tissue_indices = [1, 4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40, 41, 47, 49, 50, 53, 54, 71, 72, 73, 75, 77, 80]  # Replace with your list of indices

    process_all_episodes(base_dir, tissue_indices, output_base_dir)

