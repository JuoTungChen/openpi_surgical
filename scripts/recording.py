#!/usr/bin/env python

import rospy
import os
import cv2
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from datetime import datetime

class VideoRecorder:
    def __init__(self):
        rospy.init_node('video_recorder', anonymous=True)
        self.recording = False
        self.output_dir = os.path.join(os.path.expanduser('~'), 'recorded_videos')
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_writer = None
        self.fps = 30  # Frames per second
        self.frame_size = (960, 540)  # Default frame size, adjust as needed

        self.start_record_sub = rospy.Subscriber('/start_recording', Bool, self.record_callback, queue_size=10)
        self.image_sub = rospy.Subscriber("/jhu_daVinci/left/image_raw/compressed", 
                                            CompressedImage,  self.image_callback, queue_size=10)
        rospy.loginfo("Video Recorder Node Initialized")
    
    def record_callback(self, msg):
        if msg.data and not self.recording:
            self.start_new_recording()
        elif not msg.data and self.recording:
            self.stop_recording()

    def start_new_recording(self):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        video_path = os.path.join(self.output_dir, f"{timestamp}.mp4")
        rospy.loginfo(f"Starting recording: {video_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.frame_size)

        if not self.video_writer.isOpened():
            rospy.logerr("Failed to initialize VideoWriter!")
            self.video_writer = None
            return

        self.recording = True
        rospy.loginfo("Recording started.")

    def stop_recording(self):
        rospy.loginfo("Stopping recording.")
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False

    def image_callback(self, msg):
        if not self.recording or self.video_writer is None:
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is not None:
                if image.shape[:2] != self.frame_size:
                    image = cv2.resize(image, self.frame_size)
                
                if self.video_writer.isOpened():
                    self.video_writer.write(image)
                else:
                    rospy.logerr("VideoWriter is not open, skipping frame.")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        recorder = VideoRecorder()
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"Unhandled exception: {e}")
