#!/usr/bin/env python
import numpy as np
import cv2
import os
from datetime import datetime
import numpy as np
import time
import math
import keyboard

# for ros stuff
import rospy
from std_msgs.msg import String, Float64MultiArray, Bool, Float64
from geometry_msgs.msg import Vector3, Transform, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, JointState
from std_msgs.msg import Int32
import pandas as pd
import dynamic_reconfigure.client

class ros_topics:

  def __init__(self):
    self.bridge = CvBridge()

    # subscribers
    self.usb_camera_sub_left = rospy.Subscriber("/jhu_daVinci/left/image_raw/compressed", 
                                            CompressedImage, self.get_camera_image_left)
    self.usb_camera_sub_right = rospy.Subscriber("/jhu_daVinci/right/image_raw/compressed", 
                                            CompressedImage, self.get_camera_image_right)
    
    # endoscope imgs
    self.endo_cam_psm1_sub = rospy.Subscriber("/PSM2/endoscope_img", 
                                            Image, self.get_endo_cam_psm1)
    self.endo_cam_psm2_sub = rospy.Subscriber("/PSM1/endoscope_img", 
                                            Image, self.get_endo_cam_psm2)

    #psm1
    self.psm1_sub = rospy.Subscriber("/PSM1/measured_cp", 
                                            PoseStamped, self.get_psm1_pose)

    self.psm1_jaw_sub = rospy.Subscriber("PSM1/jaw/measured_js",
                                      JointState, self.get_psm1_jaw)

    self.psm1_rcm_sub = rospy.Subscriber("SUJ/PSM1/measured_cp", 
                                            PoseStamped, self.get_psm1_rcm_pose)

    #psm2
    self.psm2_sub = rospy.Subscriber("/PSM2/measured_cp", 
                                            PoseStamped, self.get_psm2_pose)
    
    self.psm2_jaw_sub = rospy.Subscriber("PSM2/jaw/measured_js",
                                         JointState, self.get_psm2_jaw)
    
    self.psm2_rcm_sub = rospy.Subscriber("SUJ/PSM2/measured_cp", 
                                            PoseStamped, self.get_psm2_rcm_pose)

    # ecm
    self.ecm_sub = rospy.Subscriber("/ECM/measured_cp",
                                      PoseStamped, self.get_ecm_pose)
    self.ecm_rcm_sub = rospy.Subscriber("/SUJ/ECM/measured_cp",
                                          PoseStamped, self.get_ecm_rcm_pose)
    

    self.usb_image_left = None
    self.usb_image_right = None
    self.endo_cam_psm1 = None
    self.endo_cam_psm2 = None
    self.psm1_pose = None
    self.psm1_rcm_pose = None

    self.psm2_pose = None
    self.psm2_jaw = None
    self.psm2_rcm_pose = None

    self.ecm_pose = None
    self.ecm_rcm_pose = None

  def get_camera_image_left(self,data):
    self.usb_image_left = data
  
  def get_camera_image_right(self,data):
    self.usb_image_right = data

  def get_endo_cam_psm1(self, data):
    self.endo_cam_psm1 = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')

  def get_endo_cam_psm2(self, data):
    self.endo_cam_psm2 = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')

  def get_ecm_rcm_pose(self, data):
    self.ecm_rcm_pose = data.pose

  def get_ecm_pose(self, data):
    self.ecm_pose = data.pose

  def get_psm1_pose(self, data):
    self.psm1_pose = data.pose

  def get_psm1_jaw(self, data):
    self.psm1_jaw = data.position[0]

  def get_psm1_rcm_pose(self, data):
    self.psm1_rcm_pose = data.pose

  def get_psm2_pose(self, data):
    self.psm2_pose = data.pose
  
  def get_psm2_jaw(self, data):
    self.psm2_jaw = data.position[0]

  def get_psm2_rcm_pose(self, data):
    self.psm2_rcm_pose = data.pose