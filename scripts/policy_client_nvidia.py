import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import sys
from std_msgs.msg import String, Float32MultiArray
import pandas as pd


# path_to_pi = "/home/grapes/catkin_ws/src/openpi_surgical"

# if path_to_pi:
#     sys.path.append(os.path.join(path_to_pi))

import rospy
from dvrk_scripts.dvrk_control import example_application
from dvrk_scripts.rostopics import ros_topics
from utils import set_seed # helper functions
from sklearn.preprocessing import normalize
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int16

import cv2
import crtk
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
import time
import IPython
from cv_bridge import CvBridge, CvBridgeError

e = IPython.embed
set_seed(0)

import dataclasses
import enum
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from copy import deepcopy



class EnvMode(enum.Enum):
    """Supported environments."""
    DVRK = "dvrk"


@dataclasses.dataclass
class Args:
    #host: str = "0.0.0.0"
    #host: str = "127.0.0.1"
    # host: str = "10.162.34.150"
    host: str = "10.162.34.202"
    port: int = 8000

    env: EnvMode = EnvMode.DVRK
    num_steps: int = 10
    use_stereo: bool = False
    use_6d: bool = False
    no_states: bool = False
    skip_every: int = 1
    
    
    


class LowLevelPolicy:

    ## ----------------- initializations ----------------
    def __init__(self, args):
        
        self.initialize_parameters(args)
        
        self.initialize_ros()
        
        self.obs_fn = {
            EnvMode.DVRK: self.get_observation_dvrk_stereo if self.stereo else self.get_observation_dvrk,
        }[args.env]

        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        logging.info(f"Server metadata: {self.policy.get_server_metadata()}")
        
        
    def initialize_parameters(self, args):
        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]
        
        self.header_name_actions_prsm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                    "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                    "psm1_jaw_sp"]

        self.header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                    "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                    "psm2_jaw_sp"]
        
        
        self.num_inferences = 4000
        self.action_execution_horizon = 20
        self.chunk_size = 50
        
        self.sleep_rate = 0.15
        self.state_dim = 16
        self.iter = 0
        self.sketch_img = None
        self.sketch_inferencing = False
        self.pause = False
        self.fps = 30
        self.use_contour = None
        self.correction = None
        self.user_correction = None
        self.is_correction = False
        self.user_correction_start_t = None
        self.use_preprogrammed_correction = False
        # self.rot_6d = True
        self.get_img_from_dataset = False
        self.rot_6d = args.use_6d
        self.no_states = args.no_states
        print(args.use_stereo)
        self.stereo = args.use_stereo
        self.skip_every = args.skip_every
        # self.stereo = False


    def initialize_ros(self):
        self.language_instruction = None
        self.rt = ros_topics()
        self.ral = crtk.ral('dvrk_arm_test')
        self.bridge = CvBridge()
        self.psm1_app = example_application(self.ral, "PSM1", 1)
        self.psm2_app = example_application(self.ral, "PSM2", 1)
        self.instruction_sub = rospy.Subscriber("/instructor_prediction", String, self.language_instruction_callback, queue_size=10)
        self.pause_sub = rospy.Subscriber("/pause_robot", Bool, self.pause_robot_callback, queue_size=10)
        self.action_horizon_sub = rospy.Subscriber("/action_horizon", Int16, self.action_horizon_callback, queue_size=10)
        
        #     ## --------------------- callbacks -----------------------
    def language_instruction_callback(self, msg):
        print("instruction:", msg.data)
        self.language_instruction = msg.data
    
    def pause_robot_callback(self, msg):
        self.pause = msg.data
        
        if self.pause:
            print("Robot paused. Waiting for the robot to be unpaused...")
        else:
            print("Robot unpaused. Resuming the low level policy...")
        
    def action_horizon_callback(self, msg):
        self.action_execution_horizon = msg.data
        print("action horizon changed to: ", self.action_execution_horizon)
        
        

    def main(self, args: Args) -> None:
        t = 0
        while t < self.num_inferences:
            try:
                if rospy.is_shutdown():
                    print("ROS shutdown signal received. Exiting...")
                    break
                if self.pause:
                    try:
                        time.sleep(0.2)
                    except KeyboardInterrupt:
                        print("Exiting...")
                        break
                    continue
                    
                # Send 1 observation to make sure the model is loaded.
                action_chunk = self.policy.infer(self.obs_fn())
                action = action_chunk["actions"]
                # print(action.shape)
                # input("Press Enter to continue...")
                # print(action)
                # input("Press Enter to continue...")
                
                # action = self.unnormalize_action(action, self.task_config['norm_scheme'])


                qpos_psm1 = np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                                    self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                                    self.rt.psm1_jaw))

                qpos_psm2 = np.array((self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                                    self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                                    self.rt.psm2_jaw))
                # qpos_psm1 = np.zeros((8))

                # qpos_psm2 = np.zeros((8))
                actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw

                
                if self.rot_6d == True:
                    actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
                    actions_psm1 = self.convert_delta_6d_to_taskspace_quat(action[:, 0:10], actions_psm1, qpos_psm1)
                    actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
                    actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
                    actions_psm2 = self.convert_delta_6d_to_taskspace_quat(action[:, 10:], actions_psm2, qpos_psm2)
                    actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  
                else:
                    action_normalized = deepcopy(action)
                    action_normalized[:, 0:3] -= action[0, 0:3]
                    action_normalized[:, 7:10] -= action[0, 7:10]
                    actions_psm1[:, 0:3] = qpos_psm1[0:3] + action_normalized[:, 0:3] # convert to current translation
                    actions_psm1 = self.convert_delta_rotvec_to_taskspace_quat(action[:, 0:7], actions_psm1, qpos_psm1)
                    actions_psm1[:, 7] = np.clip(action[:, 6], -0.698, 0.698)  # copy over gripper angles
                    actions_psm2[:, 0:3] = qpos_psm2[0:3] + action_normalized[:, 7:10] # convert to current translation
                    actions_psm2 = self.convert_delta_rotvec_to_taskspace_quat(action[:, 7:], actions_psm2, qpos_psm2)
                    actions_psm2[:, 7] = np.clip(action[:, 13], -0.698, 0.698)  # copy over gripper angles  
                        
                # print("actions_psm1: ", actions_psm1, "\nactions_psm2: ", actions_psm2)
                # Send actions to the robot (assume methods are implemented)
                # self.plot_actions_psm2( qpos_psm2, actions_psm2)

                # self.plot_actions_comparison(actions_psm2, self.action_psm2_gt)
                # if not self.is_correction:
                self.execute_actions(actions_psm1, actions_psm2)
                
                t += 1
                
            except KeyboardInterrupt:
                print("low level policy interrupted")
                break
        # start = time.time()
        # for _ in range(args.num_steps):
        #     policy.infer(obs_fn())
        # end = time.time()

        # print(f"Total time taken: {end - start:.2f} s")
        # print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")


    def convert_delta_6d_to_taskspace_quat(self, all_actions, all_actions_converted, qpos):
        '''
        convert delta rot into task-space quaternion rot
        '''
        # Gram-schmidt
        c1 = all_actions[:, 3:6] # t x 3
        c2 = all_actions[:, 6:9] # t x 3 
        c1 = normalize(c1, axis = 1) # t x 3
        dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
        c2 = normalize(c2 - dot_product*c1, axis = 1)
        c3 = np.cross(c1, c2)
        r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
        # transform delta rot into task space
        rots = R.from_matrix(r_mat)
        rot_init = R.from_quat(qpos[3:7])
        rots = (rot_init * rots).as_quat()
        all_actions_converted[:, 3:7] = rots
        return all_actions_converted
    

    def convert_delta_rotvec_to_taskspace_quat(self, all_actions, all_actions_converted, qpos):
        '''
        convert delta rot into task-space quaternion rot
        '''
        rot_vec = deepcopy(all_actions[:, 3:6])
        rot_diff = R.from_rotvec(rot_vec) # t x 3
        rot_init = R.from_quat(qpos[3:7])
        rots = (rot_init * rot_diff).as_quat()
        all_actions_converted[:, 3:7] = rots
        
        return all_actions_converted

    def get_observation_dvrk(self) -> dict:
        if self.get_img_from_dataset:
            dataset_path = "/home/grapes/Desktop/needle_pickup_1"
            start_ts = 36
            self.left_img = cv2.imread(dataset_path + f"/left_img_dir/frame{start_ts:06d}_left.jpg")
            lw_img = cv2.imread(dataset_path + f"/endo_psm2/frame{start_ts:06d}_psm2.jpg")
            rw_img = cv2.imread(dataset_path + f"/endo_psm1/frame{start_ts:06d}_psm1.jpg")
            ee_csv_path = os.path.join(dataset_path, "ee_csv.csv")
            ee_csv = pd.read_csv(ee_csv_path)
            # qpos_psm1 = ee_csv[self.header_name_qpos_psm1].iloc[start_ts, :].to_numpy()
            # action_psm1 = ee_csv[self.header_name_actions_psm1].iloc[start_ts:start_ts+self.chunk_size].to_numpy() 
            # qpos_psm2 = ee_csv[self.header_name_qpos_psm2].iloc[start_ts, :].to_numpy()
            self.action_psm2_gt = ee_csv[self.header_name_actions_psm2].iloc[start_ts:start_ts+self.chunk_size].to_numpy() 
            
        else:
            
            self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)
            self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
            # self.right_img = np.fromstring(self.rt.usb_image_right.data, np.uint8)
            lw_img = self.rt.endo_cam_psm2
            rw_img = self.rt.endo_cam_psm1

        self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
        self.left_img = self.left_img / 255.0
        # plt.savefig("/home/grapes/Desktop/left_img.png")
        # plt.imshow(self.left_img)
        # plt.show()

        
        # self.right_img = cv2.imdecode(self.right_img, cv2.IMREAD_COLOR)
        # self.right_img = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2RGB)
        # self.right_img = self.right_img / 255.0
        

        lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
        lw_img = lw_img / 255.0
        
        rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
        rw_img = rw_img / 255.0
        # plt.imshow(rw_img)
        # plt.show()

        if self.no_states:
            state_result = np.zeros((self.state_dim))
            # print("no states")
        else:
            state_result= np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                            self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                            self.rt.psm1_jaw,
                            self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                            self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                            self.rt.psm2_jaw))
        return {
            "state": state_result,
            "left_image": self.left_img,
            "endo_psm1_image": rw_img,
            "endo_psm2_image": lw_img,
            # "prompt": "pick up the needle and hand it to the other arm" if self.language_instruction is None else self.language_instruction,
            "prompt": "Needle Pickup" if self.language_instruction is None else self.language_instruction,
            # "prompt": "1_needle_pickup" if self.language_instruction is None else self.language_instruction,
            "actions_is_pad": np.full((self.chunk_size), False),
        }

    def get_observation_dvrk_stereo(self) -> dict:

        self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)
        self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
        self.right_img = np.fromstring(self.rt.usb_image_right.data, np.uint8)
        self.right_img = cv2.imdecode(self.right_img, cv2.IMREAD_COLOR)


        self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
        self.left_img = self.left_img / 255.0

        self.right_img = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2RGB)
        self.right_img = self.right_img / 255.0
        


        if self.no_states:
            state_result = np.zeros((self.state_dim))
        else:
            state_result= np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                            self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                            self.rt.psm1_jaw,
                            self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                            self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                            self.rt.psm2_jaw))
        return {
            "state": state_result,
            "left_image": self.left_img,
            "right_image": self.right_img,
            "prompt": "pick up the needle and hand it to the other arm" if self.language_instruction is None else self.language_instruction,
            # "prompt": "needle pickup" if self.language_instruction is None else self.language_instruction,
            # "prompt": "1_needle_pickup" if self.language_instruction is None else self.language_instruction,
            "actions_is_pad": np.full((self.chunk_size), False),
        }

    def plot_actions(self, qpos_psm1, qpos_psm2, actions_psm1, actions_psm2):
        factor = 1000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(actions_psm1[:, 0] * factor, actions_psm1[:, 1]* factor, actions_psm1[:, 2]* factor, c ='k', label = 'psm1 trajectory')
        ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='r', label = 'psm2 trajectory')
        ax.scatter(qpos_psm1[0]* factor, qpos_psm1[1]* factor, qpos_psm1[2]* factor, c = 'g', marker="*", s = 10, label = 'Current psm1 position')
        ax.scatter(qpos_psm2[0]*factor, qpos_psm2[1]*factor, qpos_psm2[2]*factor, c = 'b', marker="*" , s = 10, label = 'Current psm2 position')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        n_bins = 7
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
        plt.show()
        
    def plot_actions_psm2(self, qpos_psm2, actions_psm2):
        factor = 1000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='r', label = 'psm2 trajectory')
        ax.scatter(qpos_psm2[0]*factor, qpos_psm2[1]*factor, qpos_psm2[2]*factor, c = 'b', marker="*" , s = 10, label = 'Current psm2 position')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        n_bins = 7
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
        plt.show()

    def plot_actions_comparison(self, actions_psm2, actions_psm2_gt):
        factor = 1000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(actions_psm2_gt[:, 0] * factor, actions_psm2_gt[:, 1]* factor, actions_psm2_gt[:, 2]* factor, c ='k', label = 'gt trajectory')
        ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='r', label = 'pred trajectory')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        n_bins = 7
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
        plt.show()


    def execute_actions(self, actions_psm1, actions_psm2):

            
        for jj in range(self.action_execution_horizon):

            if not self.pause:

                self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[self.skip_every * jj])
                self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[self.skip_every * jj])
                time.sleep(self.sleep_rate)
            else:
                break
            
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    ll = LowLevelPolicy(args)
    
    ll.main(tyro.cli(Args))

