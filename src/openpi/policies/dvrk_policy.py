import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from scipy.spatial.transform import Rotation as R


def make_dvrk_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.zeros((16,)),
        "images": {
            "left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "endo_psm1": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "endo_psm2": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class DvrkInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left", "right", "endo_psm1", "endo_psm2")

    def __call__(self, data: dict) -> dict:
        data = _decode_dvrk(data, adapt_to_pi=self.adapt_to_pi)
        
        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["left"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "endo_psm2",
            "right_wrist_0_rgb": "endo_psm1",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # print(data["action"])
        if 'actions' in data:
            qpos_psm1 = data["state"][:8]
            qpos_psm2 = data["state"][8:]

            action_psm1 = data["action"][:, :8]
            action_psm2 = data["action"][:, 8:]
            # print("qpos_psm1: ", qpos_psm1)
            # print("qpos_psm2: ", qpos_psm2)
            # print("action_psm1: ", action_psm1.shape)
            # print("action_psm2: ", action_psm2.shape)

            diff_psm1 = None
            diff_psm2 = None
            # compute hybrid-relative actions. see: https://surgical-robot-transformer.github.io/
            diff_psm1 = return_actions(qpos_psm1, action_psm1)
            diff_psm2 = return_actions(qpos_psm2, action_psm2)

            # stack the actions along column dim
            diff_action = np.column_stack((diff_psm1, diff_psm2))

            # print("diff_action: ", diff_action.shape)
            # print("diff_action: ", diff_action)

        # # Actions are only available during training.
        # if "actions" in data:
            # actions = np.asarray(data["actions"])
            # actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = transforms.pad_to_dim(diff_action, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # input("Press Enter to continue...")


        return inputs


@dataclasses.dataclass(frozen=True)
class DvrkInputs_NoStates(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left", "right", "endo_psm1", "endo_psm2")

    def __call__(self, data: dict) -> dict:
        data = _decode_dvrk(data, adapt_to_pi=self.adapt_to_pi)

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(np.zeros_like(data["state"]), self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["left"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "endo_psm2",
            "right_wrist_0_rgb": "endo_psm1",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # print(data["action"])
        if 'actions' in data:
            qpos_psm1 = data["state"][:8]
            qpos_psm2 = data["state"][8:]

            action_psm1 = data["action"][:, :8]
            action_psm2 = data["action"][:, 8:]

            diff_psm1 = None
            diff_psm2 = None
            # compute hybrid-relative actions. see: https://surgical-robot-transformer.github.io/
            diff_psm1 = return_actions(qpos_psm1, action_psm1)
            diff_psm2 = return_actions(qpos_psm2, action_psm2)

            # stack the actions along column dim
            diff_action = np.column_stack((diff_psm1, diff_psm2))

            inputs["actions"] = transforms.pad_to_dim(diff_action, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # input("Press Enter to continue...")


        return inputs


@dataclasses.dataclass(frozen=True)
class DvrkOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_dvrk(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    # state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")
        return img

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions

def return_actions(qpos, action):
    """
    Computes the relative actions with respect to the current position using axis-angle rotation.

    Parameters:
    - qpos: Current pose (array of shape [8] - xyz, xyzw, jaw angle)
    - action: Actions commanded by the user (array of shape [n_actions x 8] - xyz, xyzw, jaw angle)

    Returns:
    - diff_expand: Relative actions with delta translation and delta rotation in axis-angle format.
                Shape: (n_actions, 7) - [delta_translation, delta_rotation, jaw_angle]
    """
    
    return action

def compute_diff_actions(qpos, action):
    """
    Computes the relative actions with respect to the current position using axis-angle rotation.

    Parameters:
    - qpos: Current pose (array of shape [8] - xyz, xyzw, jaw angle)
    - action: Actions commanded by the user (array of shape [n_actions x 8] - xyz, xyzw, jaw angle)

    Returns:
    - diff_expand: Relative actions with delta translation and delta rotation in axis-angle format.
                Shape: (n_actions, 7) - [delta_translation, delta_rotation, jaw_angle]
    """
    
    print("qpos: ", qpos)
    print("action: ", action)
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

def normalize_actions(diffs):
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
