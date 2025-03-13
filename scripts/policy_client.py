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
    host: str = "10.162.34.150"
    port: int = 8000

    env: EnvMode = EnvMode.DVRK
    num_steps: int = 10



class LowLevelPolicy:

    ## ----------------- initializations ----------------
    def __init__(self, args):
        
        self.initialize_parameters()
        
        self.initialize_ros()
        
        self.obs_fn = {
            EnvMode.DVRK: self.get_observation_dvrk,
        }[args.env]

        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
        logging.info(f"Server metadata: {self.policy.get_server_metadata()}")
        
        
    def initialize_parameters(self):
        self.num_inferences = 4000
        self.action_execution_horizon = 10
        
        self.sleep_rate = 0.18
        self.language_encoder = "distilbert"
        self.max_timesteps = 400 
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
        

    def initialize_ros(self):
        self.rt = ros_topics()
        self.ral = crtk.ral('dvrk_arm_test')
        self.bridge = CvBridge()
        self.psm1_app = example_application(self.ral, "PSM1", 1)
        self.psm2_app = example_application(self.ral, "PSM2", 1)
        self.instruction_sub = rospy.Subscriber("/instructor_prediction", String, self.language_instruction_callback, queue_size=10)
        self.embedding_sub = rospy.Subscriber("/instructor_embedding", Float32MultiArray, self.embeddings_callback, queue_size=10)
        self.sketch_sub = rospy.Subscriber("/sketch_output", Image, self.sketch_callback, queue_size=10)
        self.mid_level_sketch_sub = rospy.Subscriber("/mid_level_img", Image, self.mid_level_sketch_callback, queue_size=1)
        self.pause_sub = rospy.Subscriber("/pause_robot", Bool, self.pause_robot_callback, queue_size=10)
        self.action_horizon_sub = rospy.Subscriber("/action_horizon", Int16, self.action_horizon_callback, queue_size=10)
        self.contour_image_sub = rospy.Subscriber('/yolo_contour_image', Image, self.contour_image_callback, queue_size=10)
        self.use_contour_image_sub = rospy.Subscriber('/use_contour_image', Bool, self.use_contour_image_callback, queue_size=10)
        self.robot_direction_correction_sub = rospy.Subscriber("/direction_instruction", String, self.robot_direction_callback, queue_size=1)
        rospy.Subscriber('/is_correction', Bool, self.is_correction_callback, queue_size=1)
        rospy.Subscriber('/direction_instruction_user', String, self.user_correction_callback, queue_size=1)
        rospy.Subscriber("/use_preprogrammed_correction", Bool, self.use_preprogrammed_correction_callback)
        
        #     ## --------------------- callbacks -----------------------
    def language_instruction_callback(self, msg):
        self.language_instruction = msg.data

    def embeddings_callback(self, msg):
        self.language_embedding = np.array(msg.data)
        
    def sketch_callback(self, msg):
        self.sketch_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        # self.action_execution_horizon = 100
        print("sketch received")
    
    def pause_robot_callback(self, msg):
        self.pause = msg.data
        
        if self.pause:
            print("Robot paused. Waiting for the robot to be unpaused...")
        else:
            print("Robot unpaused. Resuming the low level policy...")
        
    def action_horizon_callback(self, msg):
        self.action_execution_horizon = msg.data
        print("action horizon changed to: ", self.action_execution_horizon)
        
    def contour_image_callback(self, msg):
        self.contour_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        
    def use_contour_image_callback(self, msg):
        self.use_contour = msg.data
        print("use contour image: ", self.use_contour)
        
    def robot_direction_callback(self, msg):
        self.correction = msg.data
        if self.user_correction is not None:
            if time.time() - self.user_correction_start_t < 5:
                self.correction = self.user_correction
            else:
                self.user_correction = None
                self.is_correction = False  # Reset is_correction when user correction expires

    def user_correction_callback(self, msg):
        self.user_correction = msg.data
        self.user_correction_start_t = time.time()
        self.correction = self.user_correction
        self.is_correction = True  # Set the correction flag immediately when user issues a correction
        print("User correction issued: ", self.correction)

    def is_correction_callback(self, msg):
        if self.user_correction is not None and time.time() - self.user_correction_start_t < 5:
            self.is_correction = True
        else:
            self.is_correction = msg.data
            if not self.is_correction:
                self.user_correction = None  # Clear user_correction if it's not active

    
    def mid_level_sketch_callback(self, msg):
        self.sketch_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        print("mid level sketch received")
        
        
    def use_preprogrammed_correction_callback(self, msg):
        self.use_preprogrammed_correction = msg.data
        print("use preprogrammed correction: ", self.use_preprogrammed_correction)   


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
                # action = self.unnormalize_action(action, self.task_config['norm_scheme'])


                qpos_psm1 = np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                                    self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                                    self.rt.psm1_jaw))

                qpos_psm2 = np.array((self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                                    self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                                    self.rt.psm2_jaw))

                actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw

                actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
                actions_psm1 = self.convert_delta_rotvec_to_taskspace_quat(action[:, 0:7], actions_psm1, qpos_psm1)
        
                actions_psm1[:, 7] = np.clip(action[:, 6], -0.698, 0.698)  # copy over gripper angles
                
                actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 7:10] # convert to current translation
                actions_psm2 = self.convert_delta_rotvec_to_taskspace_quat(action[:, 7:], actions_psm2, qpos_psm2)
                actions_psm2[:, 7] = np.clip(action[:, 13], -0.698, 0.698)  # copy over gripper angles  
                
                        
                # print("actions_psm1: ", actions_psm1, "\nactions_psm2: ", actions_psm2)
                # Send actions to the robot (assume methods are implemented)
                # self.plot_actions(qpos_psm1, qpos_psm2, actions_psm1, actions_psm2)
                # if not self.is_correction:
                self.execute_actions(actions_psm1, actions_psm2)
                # self.execute_actions(actions_psm1, actions_psm2, actions_psm1_temp, actions_psm2_temp)
                
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


    # def compute_diff_actions(qpos, action):
    #     """
    #     Computes the relative actions with respect to the current position using axis-angle rotation.

    #     Parameters:
    #     - qpos: Current pose (array of shape [8] - xyz, xyzw, jaw angle)
    #     - action: Actions commanded by the user (array of shape [n_actions x 8] - xyz, xyzw, jaw angle)

    #     Returns:
    #     - diff_expand: Relative actions with delta translation and delta rotation in axis-angle format.
    #                 Shape: (n_actions, 7) - [delta_translation, delta_rotation, jaw_angle]
    #     """
    #     # Compute the delta translation w.r.t da vinci endoscope tip frame (approx the camera frame)
    #     delta_translation = action[:, 0:3] - qpos[0:3]  # Shape: (n_actions, 3)

    #     # Extract quaternions from qpos and action
    #     quat_init = qpos[3:7]          # Shape: (4,)
    #     quat_actions = action[:, 3:7]  # Shape: (n_actions, 4)

    #     # Convert quaternions to Rotation objects
    #     r_init = R.from_quat(quat_init)
    #     r_actions = R.from_quat(quat_actions)

    #     # Compute the relative rotations
    #     diff_rs = r_init.inv() * r_actions  # Shape: (n_actions,)

    #     # Convert the rotation differences to rotation vectors (axis-angle representation)
    #     delta_rotation = diff_rs.as_rotvec()  # Shape: (n_actions, 3)

    #     # Extract the jaw angle from the action (note: jaw angle is not relative)
    #     jaw_angle = action[:, -1]  # Shape: (n_actions,)

    #     # Prepare the final diff array
    #     delta_action = np.zeros((action.shape[0], 7))  # Shape: (n_actions, 7)

    #     # Populate the diff_expand array
    #     delta_action[:, 0:3] = delta_translation       # Delta translation
    #     delta_action[:, 3:6] = delta_rotation          # Delta rotation (axis-angle)
    #     delta_action[:, 6] = jaw_angle                 # Jaw angle (not relative)

    #     return delta_action


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
        self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)

        self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
        self.left_img = cv2.resize(self.left_img, (224, 224))
        self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)

        self.left_img = rearrange(self.left_img, 'h w c -> c h w')
        
        self.right_img = np.fromstring(self.rt.usb_image_right.data, np.uint8)
        self.right_img = cv2.imdecode(self.right_img, cv2.IMREAD_COLOR)
        self.right_img = cv2.resize(self.right_img, (224, 224))
        self.right_img = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2RGB)
        self.right_img = rearrange(self.right_img, 'h w c -> c h w')
        
        lw_img = self.rt.endo_cam_psm2

        lw_img = cv2.resize(lw_img, (224, 224))
        lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
        lw_img = rearrange(lw_img, 'h w c -> c h w')

        rw_img = self.rt.endo_cam_psm1
        rw_img = cv2.resize(rw_img, (224, 224))
        rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
        rw_img = rearrange(rw_img, 'h w c -> c h w')
        
        
        return {
            "state": np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                            self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                            self.rt.psm1_jaw,
                            self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                            self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                            self.rt.psm2_jaw)),
            "images": {
                "left": self.left_img,
                "right": self.right_img,
                "endo_psm1": rw_img,
                "endo_psm2": lw_img,
            },
            "prompt": "1_needle_pickup",
        }


    def plot_actions(self, qpos_psm1, qpos_psm2, actions_psm1, actions_psm2):
        factor = 1000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(actions_psm1[:, 0] * factor, actions_psm1[:, 1]* factor, actions_psm1[:, 2]* factor, c ='r')
        ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='r', label = 'Generated trajectory')
        ax.scatter(qpos_psm1[0]* factor, qpos_psm1[1]* factor, qpos_psm1[2]* factor, c = 'g')
        ax.scatter(qpos_psm2[0]*factor, qpos_psm2[1]*factor, qpos_psm2[2]*factor, c = 'b', label = 'Current end-effector position')
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

            self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
            self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
            time.sleep(self.sleep_rate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ll = LowLevelPolicy(Args)
    ll.main(tyro.cli(Args))



# class LowLevelPolicy:

#     ## ----------------- initializations ----------------
#     def __init__(self, args):
#         self.args = args
#         self.temporal_agg = False
        
#         self.initialize_parameters()
#         self.initialize_ros()
        
#         self.setup_policy()
#         self.setup_language_model()
#         self.language_embedding = None
#         self.language_instruction = None
#         # self.avail_commands = ["needle pickup", "needle throw", "grabbing gallbladder", "clipping first clip left tube", "going back first clip left tube", 
#         #     "clipping second clip left tube", "going back second clip left tube",
#         #     "clipping third clip left tube", "going back third clip left tube",
#         #     "go to the cutting position left tube", "go back from the cut left tube",
#         #     "clipping first clip right tube", "going back first clip right tube",
#         #     "clipping second clip right tube", "going back second clip right tube",
#         #     "clipping third clip right tube", "going back third clip right tube",
#         #     "go to the cutting position right tube", "go back from the cut right tube",
#         #     ]
#         self.avail_commands = ["retract_home_in", "resect_home_in", "retract_home_out", "resect_home_out", "resect",  ]
#         command_idx = 0
#         self.command = self.avail_commands[command_idx] ## can change
#         # self.command = "go back" ## can change
#         self.debugging = False
        
#     def initialize_parameters(self):
#         self.num_inferences = 4000
#         self.action_execution_horizon = 30
        
#         self.sleep_rate = 0.18
#         self.language_encoder = "distilbert"
#         self.max_timesteps = 400 
#         self.state_dim = 16
#         self.iter = 0
#         self.sketch_img = None
#         self.sketch_inferencing = False
#         self.pause = False
#         self.fps = 30
#         self.use_contour = None
#         self.correction = None
#         self.user_correction = None
#         self.is_correction = False
#         self.user_correction_start_t = None
#         self.use_preprogrammed_correction = False
        
#         if self.temporal_agg:
#             self.sleep_rate = 0.05
#             self.action_execution_horizon = 1
            
#     def initialize_ros(self):
#         self.rt = ros_topics()
#         self.ral = crtk.ral('dvrk_arm_test')
#         self.bridge = CvBridge()
#         self.psm1_app = example_application(self.ral, "PSM1", 1)
#         self.psm2_app = example_application(self.ral, "PSM2", 1)
#         self.instruction_sub = rospy.Subscriber("/instructor_prediction", String, self.language_instruction_callback, queue_size=10)
#         self.embedding_sub = rospy.Subscriber("/instructor_embedding", Float32MultiArray, self.embeddings_callback, queue_size=10)
#         self.sketch_sub = rospy.Subscriber("/sketch_output", Image, self.sketch_callback, queue_size=10)
#         self.mid_level_sketch_sub = rospy.Subscriber("/mid_level_img", Image, self.mid_level_sketch_callback, queue_size=1)
#         self.pause_sub = rospy.Subscriber("/pause_robot", Bool, self.pause_robot_callback, queue_size=10)
#         self.action_horizon_sub = rospy.Subscriber("/action_horizon", Int16, self.action_horizon_callback, queue_size=10)
#         self.contour_image_sub = rospy.Subscriber('/yolo_contour_image', Image, self.contour_image_callback, queue_size=10)
#         self.use_contour_image_sub = rospy.Subscriber('/use_contour_image', Bool, self.use_contour_image_callback, queue_size=10)
#         self.robot_direction_correction_sub = rospy.Subscriber("/direction_instruction", String, self.robot_direction_callback, queue_size=1)
#         rospy.Subscriber('/is_correction', Bool, self.is_correction_callback, queue_size=1)
#         rospy.Subscriber('/direction_instruction_user', String, self.user_correction_callback, queue_size=1)
#         rospy.Subscriber("/use_preprogrammed_correction", Bool, self.use_preprogrammed_correction_callback)
        
        
#     def setup_policy(self):
#         self.task_config = TASK_CONFIGS[self.args.task_name]
#         self.mean = self.task_config['action_mode'][1]['mean']
#         self.std = self.task_config['action_mode'][1]['std']
#         self.max_ = self.task_config['action_mode'][1]['max_']
#         self.min_ = self.task_config['action_mode'][1]['min_']
#         self.action_mode = self.task_config['action_mode'][0]
#         if self.task_config.get('use_history'):
#             self.use_history = self.task_config['use_history']
#         else:
#             self.use_history = False
            
#         if self.task_config.get('use_sketch'):
#             self.use_sketch = self.task_config['use_sketch']
#         else:
#             self.use_sketch = False
            
#         if self.task_config.get('use_auto_label'):
#             self.use_auto_label = self.task_config['use_auto_label']
#         else:
#             self.use_auto_label = False
            
#         if self.use_auto_label:
#             self.chunk_size = 60
#         else:
#             self.chunk_size = 60
            
            
#         if self.task_config.get('estimation'):
#             self.estimation = True
#             print("\n\nusing estimation!!\n")
#         else:
#             self.estimation = False
            
#         self.num_queries = self.chunk_size
        
#         self.left_img = None
#         if self.use_history:
#             self.left_img_hist = None

#         if self.args.policy_class == "ACT":
#             policy_config = {
#                 'lr': 1e-5,
#                 'num_queries': self.chunk_size,
#                 'action_dim': 20,
#                 'kl_weight': 10,
#                 'hidden_dim': 512,
#                 'dim_feedforward': 3200,
#                 'lr_backbone': 1e-5,
#                 'backbone': 'efficientnet_b3' if not self.args.use_language else "efficientnet_b3film",
#                 'enc_layers': 4,
#                 'num_epochs': self.args.num_epochs,
#                 'dec_layers': 7,
#                 'nheads': 8,
#                 'camera_names': self.task_config['camera_names'],
#                 "multi_gpu": None,
#             }
#             self.policy = ACTPolicy(policy_config)
            
#         elif self.args.policy_class == "Diffusion":
#             policy_config = {
#                 "lr": 1e-5,
#                 'camera_names': self.task_config['camera_names'],
#                 "action_dim": 20,
#                 "observation_horizon": 1,
#                 "action_horizon": 8,  # TODO not used
#                 "prediction_horizon": self.chunk_size,
#                 "num_queries": self.chunk_size,
#                 "num_inference_timesteps": 50,
#                 "ema_power": 0.75,
#                 "vq": False,
#                 "backbone": 'resnet18film',   # efficientnet_b3film or resnet18film
#                 "multi_gpu": False,
#                 "is_eval": True,
#             }
#             self.policy = DiffusionPolicyNoSpatialSoftmax(policy_config)
            
#         # checkpoint = torch.load(self.args.ckpt_dir)
#         # self.policy.deserialize(checkpoint['model_state_dict'])
#         # self.policy.cuda()
#         # self.policy.eval()
        
#         model_state_dict = torch.load(self.args.ckpt_dir)["model_state_dict"]
#         if is_multi_gpu_checkpoint(model_state_dict):
#             print("The checkpoint was trained on multiple GPUs.")
#             model_state_dict = {
#                 k.replace("module.", "", 1): v for k, v in model_state_dict.items()
#             }
#         loading_status = self.policy.deserialize(model_state_dict)
#         print(loading_status)
#         self.policy.cuda()
#         self.policy.eval()
        
#         # print(f"Loaded: {self.args.ckpt_dir}")
    
#     def setup_language_model(self):
#         if self.args.use_language:
#             self.tokenizer, self.model = initialize_model_and_tokenizer("distilbert")
#             assert self.tokenizer is not None and self.model is not None
#             print("language model and tokenizer set up completed")
    


#     ## ------------ helper functions for action -------------
#     def convert_6d_rot_to_quat(self, rots):
#         c1 = rots[:, 0:3]
#         c2 = rots[:, 3:6]
#         c1 = normalize(c1, axis=1)
#         dot_product = np.sum(c1 * c2, axis=1).reshape(-1, 1)
#         c2 = normalize(c2 - dot_product * c1, axis=1)
#         c3 = np.cross(c1, c2)
#         r_mat = np.dstack((c1, c2, c3))
#         rots = R.from_matrix(r_mat)
#         return rots.as_quat()

#     def convert_actions_to_SE3_then_final_actions(self, dts, dquats, qpos_psm, jaw_angles):
#         dquats = batch_rotations.batch_quaternion_wxyz_from_xyzw(dquats)
#         qpos_psm[3:7] = rotations.quaternion_wxyz_from_xyzw(qpos_psm[3:7])
#         dts_dquats = np.concatenate((dts, dquats), axis=1)
#         g_qpos = transformations.transform_from_pq(qpos_psm[0:7])
#         g_actions = trajectories.transforms_from_pqs(dts_dquats)
#         g_poses = trajectories.concat_one_to_many(g_qpos, g_actions)
#         output = np.zeros((dquats.shape[0], 8))
#         output[:, 0:3] = g_poses[:, 0:3, 3]
#         tmp = batch_rotations.quaternions_from_matrices(g_poses[:, 0:3, 0:3])
#         output[:, 3:7] = batch_rotations.batch_quaternion_xyzw_from_wxyz(tmp)
#         output[:, 7] = np.clip(jaw_angles, -0.698, 1.4)
#         return output

#     def compute_diff_actions_relative_endoscope(self, qpos, action):
#         """
#         qpos: current position [9]
#         action: actions commanded by the user [n_actions x 9]
#         returns: relative actions w.r.t qpos
#         """
#         # find diff first and then fill-in the quaternion differences properly
#         diff = action - qpos
#         quat_actions = action[:, 3:7]

#         r_actions = R.from_quat(quat_actions)
#         diff_rs = r_actions 
#         # extract their first two columns
#         diff_6d = diff_rs.as_matrix()[:,:,:2]
#         diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
#         diff_expand = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
#         diff_expand[:diff.shape[0], 0:diff.shape[1]] = diff 
#         diff = diff_expand

#         diff[:, 3:9] = diff_6d
#         diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
#         return diff

#     def unnormalize_action(self, naction, norm_scheme):
#         action = None
#         if norm_scheme == "min_max":
#             action = (naction + 1) / 2 * (self.max_ - self.min_) + self.min_
#             action[:, 3:9] = naction[:, 3:9]
#             action[:, 13:19] = naction[:, 13:19]
#         elif norm_scheme == "std":
#             action = self.unnormalize_positions_only_std(naction)
#         else:
#             raise NotImplementedError
#         return action

#     def unnormalize_positions_only_std(self, diffs):
#         unnormalized = diffs * self.std + self.mean
#         unnormalized[:, 3:9] = diffs[:, 3:9]
#         unnormalized[:, 13:19] = diffs[:, 13:19]
#         return unnormalized

#     def convert_delta_6d_to_taskspace_quat(self, all_actions, all_actions_converted, qpos):
#         '''
#         convert delta rot into task-space quaternion rot
#         '''
#         # Gram-schmidt
#         c1 = all_actions[:, 3:6] # t x 3
#         c2 = all_actions[:, 6:9] # t x 3 
#         c1 = normalize(c1, axis = 1) # t x 3
#         dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
#         c2 = normalize(c2 - dot_product*c1, axis = 1)
#         c3 = np.cross(c1, c2)
#         r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
#         # transform delta rot into task space
#         rots = R.from_matrix(r_mat)
#         rot_init = R.from_quat(qpos[3:7])
#         rots = (rot_init * rots).as_quat()
#         all_actions_converted[:, 3:7] = rots
#         return all_actions_converted
    
    
#     def convert_delta_6d_to_taskspace_quat_relative_endo(self, all_actions, all_actions_converted, qpos):
#         '''
#         convert delta rot into task-space quaternion rot
#         '''
#         # Gram-schmidt
#         c1 = all_actions[:, 3:6] # t x 3
#         c2 = all_actions[:, 6:9] # t x 3 
#         c1 = normalize(c1, axis = 1) # t x 3
#         dot_product = np.sum(c1 * c2, axis = 1).reshape(-1, 1)
#         c2 = normalize(c2 - dot_product*c1, axis = 1)
#         c3 = np.cross(c1, c2)
#         r_mat = np.dstack((c1, c2, c3)) # t x 3 x 3
#         # transform delta rot into task space
#         rots = R.from_matrix(r_mat).as_quat()
#         # rot_init = R.from_quat(qpos[3:7])
#         # rots = (rot_init * rots).as_quat()
#         all_actions_converted[:, 3:7] = rots
#         return all_actions_converted
    
#     def temporal_ensemble(self, all_time_actions, t, actions_psm1, actions_psm2):
#         # Apply temporal assembling to both actions_psm1 and actions_psm2
#         # for actions in [actions_psm1, actions_psm2]:
#         # Temporal assembling logic for positions
#         actions_for_curr_step = all_time_actions[:, t] # Obtain actions for current time step
#         # print(actions_for_curr_step.shape)
#         # print(actions_for_curr_step)
        
#         actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
#         actions_for_curr_step = actions_for_curr_step[actions_populated]
#         # print(actions_for_curr_step.shape)
#         # print(actions_for_curr_step)
#         # Exponential weights calculation
#         k = 0.05
#         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))

#         exp_weights = exp_weights / exp_weights.sum()

#         exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
#         # print(exp_weights)
#         # Apply temporal assembling to position
#         # print(actions_for_curr_step[:, :3])
#         # print(exp_weights)
#         # print(actions_for_curr_step[:, :3] * exp_weights)
#         psm1_position = (actions_for_curr_step[:, :3] * exp_weights).sum(dim=0, keepdim=True)
#         psm2_position = (actions_for_curr_step[:, 8:11] * exp_weights).sum(dim=0, keepdim=True)
#         # print(psm1_position.shape)
#         # print(psm1_position)
#         # Handle quaternion averaging
#         psm1_quaternions = actions_for_curr_step[:, 3:7]
#         psm2_quaternions = actions_for_curr_step[:, 11:15]
#         psm1_quaternion_avg = self.average_quaternions(psm1_quaternions, exp_weights)
#         psm2_quaternion_avg = self.average_quaternions(psm2_quaternions, exp_weights)

#         # Assign assembled actions back to actions_psm
#         actions_psm1[:, :3] = psm1_position.cpu().numpy()
#         actions_psm1[:, 3:7] = psm1_quaternion_avg
#         actions_psm2[:, :3] = psm2_position.cpu().numpy()
#         actions_psm2[:, 3:7] = psm2_quaternion_avg
#         return actions_psm1, actions_psm2

#     def average_quaternions(self, quaternions, weights):
#         """
#         Average a set of quaternions using weighted averaging.
#         """
#         if isinstance(weights, torch.Tensor):
#             weights = weights.cpu().numpy()  # Move to CPU and convert to NumPy
#             quaternions = quaternions.cpu().numpy()
#         # Normalize weights to sum to 1
#         weights = np.array(weights, dtype=np.float64)
#         weights /= weights.sum()

#         # Quaternion averaging: weighted mean in quaternion space
#         avg_quat = np.zeros((4,))
#         for i, quat in enumerate(quaternions):
#             avg_quat += weights[i] * quat

#         # Normalize the resulting quaternion
#         avg_quat /= np.linalg.norm(avg_quat)
#         return avg_quat
    
    
#     ## --------------------- callbacks -----------------------
#     def language_instruction_callback(self, msg):
#         self.language_instruction = msg.data

#     def embeddings_callback(self, msg):
#         self.language_embedding = np.array(msg.data)
        
#     def sketch_callback(self, msg):
#         self.sketch_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
#         # self.action_execution_horizon = 100
#         print("sketch received")
    
#     def pause_robot_callback(self, msg):
#         self.pause = msg.data
        
#         if self.pause:
#             print("Robot paused. Waiting for the robot to be unpaused...")
#         else:
#             print("Robot unpaused. Resuming the low level policy...")
        
#     def action_horizon_callback(self, msg):
#         self.action_execution_horizon = msg.data
#         print("action horizon changed to: ", self.action_execution_horizon)
        
        
#     def contour_image_callback(self, msg):
#         self.contour_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        
#     def use_contour_image_callback(self, msg):
#         self.use_contour = msg.data
#         print("use contour image: ", self.use_contour)
        
#     def robot_direction_callback(self, msg):
#         self.correction = msg.data
#         if self.user_correction is not None:
#             if time.time() - self.user_correction_start_t < 5:
#                 self.correction = self.user_correction
#             else:
#                 self.user_correction = None
#                 self.is_correction = False  # Reset is_correction when user correction expires

#     def user_correction_callback(self, msg):
#         self.user_correction = msg.data
#         self.user_correction_start_t = time.time()
#         self.correction = self.user_correction
#         self.is_correction = True  # Set the correction flag immediately when user issues a correction
#         print("User correction issued: ", self.correction)

#     def is_correction_callback(self, msg):
#         if self.user_correction is not None and time.time() - self.user_correction_start_t < 5:
#             self.is_correction = True
#         else:
#             self.is_correction = msg.data
#             if not self.is_correction:
#                 self.user_correction = None  # Clear user_correction if it's not active

#         # print("Is correction active: ", self.is_correction)
    
#     def mid_level_sketch_callback(self, msg):
#         self.sketch_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
#         print("mid level sketch received")
        
        
#     def use_preprogrammed_correction_callback(self, msg):
#         self.use_preprogrammed_correction = msg.data
#         print("use preprogrammed correction: ", self.use_preprogrammed_correction)    
    
#     ## ---------------------- helpers ------------------------
#     def get_image_dvrk_dataset(self, t=0):
#         base_dir = "/home/grapes/catkin_ws/src/rqt_mypkg/src/rqt_mypkg/_recordings/20250120-165154-008030"
#         # base_dir = "/home/grapes/catkin_ws/src/rqt_mypkg/src/rqt_mypkg/_recordings/base_chole_clipping_cutting/tissue_54/2_clipping_first_clip_left_tube_recovery/20240807-135749-215264_recovery"
#         # base_dir = "/home/grapes/catkin_ws/src/rqt_mypkg/src/rqt_mypkg/_recordings/base_chole_clipping_cutting/tissue_54/2_clipping_first_clip_left_tube/20240807-135116-990212"
#         image_idx = 700 + t * 30      # the index of the image in the dataset
#         left_img = cv2.imread(os.path.join(base_dir, "left_img_dir", f"frame000{image_idx}_left.jpg"))
#         left_img = cv2.resize(left_img, (480, 360))
#         left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
#         plt.imshow(left_img)
#         plt.show()
#         left_img = rearrange(left_img, 'h w c -> c h w')

#         lw_img = cv2.imread(os.path.join(base_dir, "endo_psm2", f"frame000{image_idx}_psm2.jpg"))
#         lw_img = cv2.resize(lw_img, (480, 360))
#         lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
#         lw_img = rearrange(lw_img, 'h w c -> c h w')

#         rw_img = cv2.imread(os.path.join(base_dir, "endo_psm1", f"frame000{image_idx}_psm1.jpg"))
#         rw_img = cv2.resize(rw_img, (480, 360))
#         rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
#         rw_img = rearrange(rw_img, 'h w c -> c h w')
#         if self.use_sketch:
#             sketch_img = np.zeros_like(left_img)
    
#             curr_image = np.stack([left_img, lw_img, rw_img, sketch_img], axis=0)
#             curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
#         else:
#             curr_image = np.stack([left_img, lw_img, rw_img], axis=0)
#             curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
            
#         return curr_image
    
#     def get_image_dvrk(self):
#         self.iter += 1
        
#         if self.use_history:
#             fig, axs = plt.subplots(1, 2)
#             if self.left_img_hist is None and self.left_img is None:

#                 self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)

                
#                 self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
#                 self.left_img = cv2.resize(self.left_img, (480, 360))
#                 self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
#                 # axs[0].imshow(self.left_img)

#                 self.left_img = rearrange(self.left_img, 'h w c -> c h w')
#                 self.left_img_hist = np.zeros_like(self.left_img)
#                 # axs[1].imshow(self.left_img_hist)
                
#             # elif self.iter == 30:
#             else:

#                 self.left_img_hist = self.left_img
                
#                 self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)


#                 self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
#                 self.left_img = cv2.resize(self.left_img, (480, 360))
#                 self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
#                 axs[0].imshow(self.left_img)
#                 axs[1].imshow(rearrange(self.left_img_hist, 'c h w -> h w c'))

#                 self.left_img = rearrange(self.left_img, 'h w c -> c h w')
                
                
#             ## subplot two column

            
#             ## show plot
#             plt.show()
#                 # self.iter = 0
#         if self.use_contour and self.contour_img is not None:
#             self.left_img = self.contour_img
#             self.left_img = cv2.resize(self.left_img, (480, 360))
#             self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
#             # plt.imshow(self.left_img)
#             # plt.show()
#             # self.left_img = rearrange(self.left_img, 'h w c -> c h w')
        
#         else:
#             # rt = ros_topics()
#             self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)
#             self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
#             self.left_img = cv2.resize(self.left_img, (480, 360))
#             self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
        
#         # new_hist_img = rearrange(self.left_img_hist, 'c h w -> h w c')
#         # stacked_img = cv2.hconcat([self.left_img, new_hist_img])
        
#         # # cv2.imshow("left_img", stacked_img)
#         # plt.imshow(stacked_img)      
#         # plt.show()
#         new_left_img = self.left_img
          
#         self.left_img = rearrange(self.left_img, 'h w c -> c h w')

#         # lw_img = self.rt.endo_cam_psm2
        

        
#         ## TODO change back to using real psm2 image
#         # image_idx = 0
#         # base_dir = "/home/grapes/catkin_ws/src/rqt_mypkg/src/rqt_mypkg/_recordings/20250226_tissue_lift/20250226-123022-358155"
        
#         # lw_img = cv2.imread(os.path.join(base_dir, "endo_psm2", f"frame000001_psm2.jpg"))
#         lw_img = self.rt.endo_cam_psm2

#         lw_img = cv2.resize(lw_img, (480, 360))
#         lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
#         lw_img = rearrange(lw_img, 'h w c -> c h w')

#         rw_img = self.rt.endo_cam_psm1
#         # plt.imshow(rw_img)
#         # plt.show()
#         # assert False
#         rw_img = cv2.resize(rw_img, (480, 360))
#         rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
#         rw_img = rearrange(rw_img, 'h w c -> c h w')

#         if self.use_history:
#             curr_image = np.stack([self.left_img, lw_img, rw_img, self.left_img_hist], axis=0)
#             curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
            
#         elif self.use_sketch:
#             if self.sketch_img is not None:
#                 self.sketch_inferencing = True
#                 print("Shape of self.sketch_img:", self.sketch_img.shape)
#                 sketch_img = None
#                 if len(self.sketch_img.shape) == 2:  # Grayscale image
#                     sketch_img = cv2.cvtColor(self.sketch_img, cv2.COLOR_GRAY2BGR)
#                 elif self.sketch_img.shape[2] == 4:  # RGBA image
#                     sketch_img = cv2.cvtColor(self.sketch_img, cv2.COLOR_RGBA2BGR)
#                 sketch_img = cv2.resize(self.sketch_img, (480, 360))
#                 sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
#                 # concat_img = cv2.hconcat([new_left_img, sketch_img])
#                 # plt.imshow(concat_img)
#                 # plt.show()
#                 sketch_img = rearrange(sketch_img, 'h w c -> c h w')

#             else:
#                 sketch_img = np.zeros_like(self.left_img)
    
#             curr_image = np.stack([self.left_img, lw_img, rw_img, sketch_img], axis=0)
#             curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
            
#             # reset sketch image
#             if self.iter % 3 == 0:
#                 self.sketch_img = None
#                 print("sketch erased")
#         else:
#             curr_image = np.stack([self.left_img, lw_img, rw_img], axis=0)
#             curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
            
#         return curr_image
    
    
#     def generate_command_embedding(self, command, correction):
#         ## use language embeddings from high level policy rostopic
#         # if self.args.use_language and self.language_embedding is not None:
#         #     command_embedding = torch.tensor(self.language_embedding).float().cuda()
#         #     print("using high level policy embeddings")
#         #     return command_embedding

#         ## use language instructions from high level policy rostopic
#         if self.args.use_language and self.language_instruction is not None:
#             # print(self.is_correction, correction)
#             # if correction is not None and correction != "do not move":
            
#             if self.use_auto_label:
#                 if self.is_correction and correction is not None and correction != "do not move":
#                     command = correction
#                 else:
#                     command = self.language_instruction
#             else:
#                 command = self.language_instruction
                
#             print(command)
#             command_embedding = encode_text(command, self.language_encoder, self.tokenizer, self.model)
#             command_embedding = torch.tensor(command_embedding).cuda()
#             print(f"\n---------------------------------------\nusing high level policy command: {command}\n---------------------------------------\n")

#             return command_embedding
        
#         ## use language command set in the low level policy
#         else:
#             if self.use_auto_label:
#                 if self.is_correction and correction is not None and correction != "no command":
#                     command = correction
#                 # command = f"{command} {correction}"
                
#             command_embedding = encode_text(command, self.language_encoder, self.tokenizer, self.model)
#             command_embedding = torch.tensor(command_embedding).cuda()
#             print(f"\n---------------------------------------\nusing command: {command}\n---------------------------------------\n")
#             return command_embedding
    
#     def plot_actions(self, qpos_psm1, qpos_psm2, actions_psm1, actions_psm2):
#         factor = 1000
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.scatter(actions_psm1[:, 0] * factor, actions_psm1[:, 1]* factor, actions_psm1[:, 2]* factor, c ='r')
#         ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='r', label = 'Generated trajectory')
#         ax.scatter(qpos_psm1[0]* factor, qpos_psm1[1]* factor, qpos_psm1[2]* factor, c = 'g')
#         ax.scatter(qpos_psm2[0]*factor, qpos_psm2[1]*factor, qpos_psm2[2]*factor, c = 'b', label = 'Current end-effector position')
#         ax.set_xlabel('X (mm)')
#         ax.set_ylabel('Y (mm)')
#         ax.set_zlabel('Z (mm)')
#         n_bins = 7
#         ax.legend()
#         ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
#         ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
#         ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
#         plt.show()
#         # input("Press Enter to continue...")
#         # assert(False)

#     def execute_actions(self, actions_psm1, actions_psm2):
#         # if self.sketch_inferencing:
#         #     print("sketch inferencing")
#         #     for jj in range(100):
#         #         self.sketch_inferencing = False
                
#         #         # print("actions_psm2: ", actions_psm2[jj], "\nactions_psm2_temp: ", actions_psm2_temp[jj])
#         #         if not self.pause:
#         #             self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
#         #             self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
#         #             time.sleep(self.sleep_rate)
#         #         else:
#         #             break
#         # # print(actions_psm1_temp.shape, actions_psm2_temp.shape) 
#         # else:
#         # if self.language_instruction == "grabbing gallbladder":
#         #     self.action_execution_horizon = 30
#         # else:
#         #     self.action_execution_horizon = 18
            
#         for jj in range(self.action_execution_horizon):
#             # print("actions_psm2: ", actions_psm2[jj], "\nactions_psm2_temp: ", actions_psm2_temp[jj])
#             # if not self.pause and not self.is_correction:
#             if not self.pause:
#                 if self.use_preprogrammed_correction and self.is_correction:
#                     break

#                 self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
#                 self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
#                 time.sleep(self.sleep_rate)
#             else:
#                 break
                

#     ## --------------------- main loop -----------------------

#     def run(self):
#         print("-------------starting low level policy------------------\n")
#         time.sleep(1)
#         if self.temporal_agg:
#             all_time_actions = torch.zeros(
#                 [self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim]
#             ).cuda()

#         with torch.inference_mode():
#             t = 0
            
#             while t < self.num_inferences:
#                 try:
#                     if rospy.is_shutdown():
#                         print("ROS shutdown signal received. Exiting...")
#                         break
                    
#                     # assert command_embedding is not None
#                     if self.pause or (self.use_preprogrammed_correction and self.is_correction):
#                     # if self.pause or self.is_correction or (self.use_preprogrammed_correction and self.is_correction):
#                         try:
#                             time.sleep(0.2)
#                         except KeyboardInterrupt:
#                             print("Exiting...")
#                             break
#                         continue
                    
#                     if self.args.use_language:
#                         command_embedding = self.generate_command_embedding(self.command, self.correction)
#                     else:
#                         command_embedding = None
#                     qpos_zero = torch.zeros(1, 20).float().cuda()
                    
#                     ## use this if testing with dataset image
#                     # curr_image = self.get_image_dvrk_dataset(t)
                    
#                     ### use this if testing with real endoscope image
#                     curr_image = self.get_image_dvrk()

                
#                     action = self.policy(qpos_zero, curr_image, command_embedding=command_embedding).cpu().numpy().squeeze()
#                     action = self.unnormalize_action(action, self.task_config['norm_scheme'])


#                     qpos_psm1 = np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
#                                         self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
#                                         self.rt.psm1_jaw))

#                     qpos_psm2 = np.array((self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
#                                         self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
#                                         self.rt.psm2_jaw))

#                     if self.action_mode == 'hybrid':
#                         # scale = 5
#                         # scale_x = 1.6
#                         # scale_y = 1.8
#                         actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
#                         # if self.estimation:
#                         #     action[:, 9] = scale * (action[:, 9] + 0.08) # scale gripper                                
#                         #     action[:, 0] = scale_x * action[:, 0] # scale x
#                         #     action[:, 1] = np.clip(action[:, 1], -0.0015, 0.0015) # clip y
#                         #     action[:, 2] = np.clip(action[:, 2], -0.001, 0.001) # clip the z-axis
                        
#                         actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
#                         actions_psm1 = self.convert_delta_6d_to_taskspace_quat(action[:, 0:10], actions_psm1, qpos_psm1)
#                         # if self.estimation:
#                         #     actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
#                         #     action[:, 2] = np.clip(action[:, 2], -0.001, 0.001) # clip the z-axis
#                         # else:
#                         actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
#                         # print("actions_psm1: ", actions_psm1)
                        
#                         actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
#                         actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
#                         actions_psm2 = self.convert_delta_6d_to_taskspace_quat(action[:, 10:], actions_psm2, qpos_psm2)
#                         actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  

#                     if self.action_mode == 'relative_endoscope':
#                         actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
#                         actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
#                         actions_psm1 = self.convert_delta_6d_to_taskspace_quat_relative_endo(action[:, 0:10], actions_psm1, qpos_psm1)
#                         actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
                        
#                         actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
#                         actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
#                         actions_psm2 = self.convert_delta_6d_to_taskspace_quat_relative_endo(action[:, 10:], actions_psm2, qpos_psm2)
#                         actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  
                        
#                     if self.action_mode == 'ego':
#                         # compute actions for PSM1
#                         actions_psm1 = np.zeros((self.chunk_size, 8)) # pos (3), quat (4), jaw (1) 
#                         dts_psm1 = action[:, 0:3]
#                         dquats_psm1 = self.convert_6d_rot_to_quat(action[:, 3:9]) # [n x 4] xyzw convention
#                         actions_psm1 = self.convert_actions_to_SE3_then_final_actions(dts_psm1, dquats_psm1, qpos_psm1, action[:, 9]) # translation and quaternion
#                         # actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
                        
#                         # compute actions for PSM2
#                         actions_psm2 = np.zeros((self.chunk_size, 8)) # pos (3), quat (4), jaw (1) 
#                         dts_psm2 = action[:, 10:13]
#                         dquats_psm2 = self.convert_6d_rot_to_quat(action[:, 13:19]) # [n x 4] xyzw convention
#                         actions_psm2 = self.convert_actions_to_SE3_then_final_actions(dts_psm2, dquats_psm2, qpos_psm2, action[:, 19]) # translation and quaternion   
#                         # actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  
                    
#                     if self.temporal_agg:
#                         # Convert lists to tensors
#                         actions_psm1_tensor = torch.tensor(actions_psm1, dtype=torch.float32, device='cuda')
#                         actions_psm2_tensor = torch.tensor(actions_psm2, dtype=torch.float32, device='cuda')
#                         # Concatenate actions along the feature dimension (dim=1)
#                         combined_actions = torch.cat([actions_psm1_tensor, actions_psm2_tensor], dim=1)  # Shape: [100, 16]

#                         # Add a new dimension to match the shape for assignment
#                         combined_actions = combined_actions.unsqueeze(0)  # Shape: [1, 100, 16]                        print(combined_actions.shape)
#                         all_time_actions[[t], t : t + self.num_queries] = combined_actions
#                         actions_psm1, actions_psm2 = self.temporal_ensemble(all_time_actions, t, actions_psm1, actions_psm2)
                        
                        
                        
#                     # print("actions_psm1: ", actions_psm1, "\nactions_psm2: ", actions_psm2)
#                     # Send actions to the robot (assume methods are implemented)
#                     self.plot_actions(qpos_psm1, qpos_psm2, actions_psm1, actions_psm2)
#                     # if not self.is_correction:
#                     self.execute_actions(actions_psm1, actions_psm2)
#                     # self.execute_actions(actions_psm1, actions_psm2, actions_psm1_temp, actions_psm2_temp)
                    
#                     if self.debugging:
#                         key = input("press enter to continue...")
#                         if key == "q":
#                             exit
#                     t += 1
                    
#                 except KeyboardInterrupt:
#                     print("low level policy interrupted")
#                     break


# ## --------------------- main function -----------------------

# if __name__ == "__main__":
#     os.environ["PYTHONWARNINGS"] = "ignore"
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ckpt_dir', action='store', type=str, 
#                         help='specify ckpt file path', 
#                         required=True)
#     # needed to avoid error for detr
#     parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
#     parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
#     parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
#     parser.add_argument('--use_language', action='store_true')
#     parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)    
#     # parser.add_argument('--command_idx', action='store', type=int, help='command_idx')    
#     args = parser.parse_args()
        
#     # rospy.init_node('dvrk_low_level_policy')
#     system = LowLevelPolicy(args)
#     system.run()
