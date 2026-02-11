import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from PyQt5.QtWidgets import QCheckBox, QButtonGroup, QRadioButton, QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QColorDialog, QSlider, QHBoxLayout, QLineEdit, QComboBox, QFrame, QCheckBox, QButtonGroup, QRadioButton, QButtonGroup, QButtonGroup
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QIcon, QCursor
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, QPointF, QTimer
from dvrk_scripts.rostopics import ros_topics
import numpy as np
from std_msgs.msg import Bool, Float32, Int16, String, Float32MultiArray
import cv2
# from auto_label_func import get_all_auto_labels_list

RADIUS = 3
LINEWIDTH = 6

class ImageSubscriber:
    def __init__(self):
        self.rt = ros_topics()

        self.bridge = CvBridge()
        self.image = None
        self.image_psm1 = None
        self.image_psm2 = None
        self.yolo_image = None
        self.use_contour_image = False
        self.mid_level_img_received = False
        self.last_mid_level_img_time = None
        self.mid_level_img_timeout = 2  # seconds to wait before considering mid_level_img stale
        rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, self.callback, queue_size=10)
        # rospy.Subscriber("/PSM1/endoscope_img", Image, self.rw_callback, queue_size=10)
        # rospy.Subscriber("/PSM2/endoscope_img", Image, self.lw_callback, queue_size=10)
        rospy.Subscriber("/mid_level_img", Image, self.mid_level_img_callback, queue_size=10)
        
        rospy.Subscriber("/yolo_contour_image", Image, self.yolo_callback, queue_size=10)
        rospy.Subscriber('/use_contour_image', Bool, self.use_contour_image_callback, queue_size=10)
        
        # Start a timer to periodically update the UI
        # self.timer = QTimer()
        # self.timer.timeout.connect(self.image_updated)
        # self.timer.start(100)  # Update every 100ms

    def callback(self, data):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # self.image = cv_image
            current_time = rospy.get_time()
            
            # self.image = np.fromstring(self.rt.usb_image_left.data, np.uint8)
            # self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
            # self.image = cv2.resize(self.image, (960, 540))
            # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            if self.mid_level_img_received and (current_time - self.last_mid_level_img_time) < self.mid_level_img_timeout:
                # Use the most recent mid_level_img
                cv_image = self.mid_level_img
            else:
                # Use the left_img as fallback
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                cv_image = cv2.resize(cv_image, (960, 540))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            self.image = cv_image
            
            
            if not self.use_contour_image:
                self.image_updated()

        except CvBridgeError as e:
            print(e)
            
    def mid_level_img_callback(self, data):
        try:
            # Store the mid_level_img and update the timestamp
            self.mid_level_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.mid_level_img = cv2.resize(self.mid_level_img, (960, 540))
            self.mid_level_img = cv2.cvtColor(self.mid_level_img, cv2.COLOR_BGR2RGB)
            self.mid_level_img_received = True
            self.last_mid_level_img_time = rospy.get_time()

            if not self.use_contour_image:
                self.image = self.mid_level_img
                self.image_updated()
                
        except CvBridgeError as e:
            print(e)
    # def rw_callback(self, data):
    #     self.image_psm1 = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')

    #     self.image_psm1 = cv2.resize(self.image_psm1, (320, 240))
    #     self.image_psm1 = cv2.cvtColor(self.image_psm1, cv2.COLOR_BGR2RGB)

    #     self.right_wrist_images_updated()

    # def lw_callback(self, data):
    #     self.image_psm2 = self.bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')

    #     self.image_psm2 = cv2.resize(self.image_psm2, (320, 240))
    #     self.image_psm2 = cv2.cvtColor(self.image_psm2, cv2.COLOR_BGR2RGB)
    #     self.left_wrist_images_updated()

    def yolo_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.yolo_image = cv_image
            self.yolo_image = cv2.resize(self.yolo_image, (960, 540))
            self.yolo_image = cv2.cvtColor(self.yolo_image, cv2.COLOR_BGR2RGB)
            
            if self.use_contour_image:
                self.image_updated()  # Notify the GUI to update the image display
        except CvBridgeError as e:
            print(e)

    def use_contour_image_callback(self, data):
        self.use_contour_image = data.data
    
    
    # def left_wrist_images_updated(self):
    #     if hasattr(self, 'update_left_wrist_images') and self.image_psm2 is not None:
        
    #         self.update_left_wrist_images(self.image_psm2)
            
    # def right_wrist_images_updated(self):
    #     if hasattr(self, 'update_right_wrist_images') and self.image_psm1 is not None:
        
    #         self.update_right_wrist_images(self.image_psm1)
    
    def image_updated(self):
        if hasattr(self, 'update_image'):
            if self.use_contour_image and self.yolo_image is not None:
                self.update_image(self.yolo_image)
            elif self.image is not None:
                self.update_image(self.image)  # Call the GUI's method to update the image
            
class ImagePublisher:
    def __init__(self):
        # Initialize the ROS publisher
        self.publisher = rospy.Publisher('/sketch_output', Image, queue_size=10)
        self.bridge = CvBridge()

    def publish_image(self, qimage):
        # Convert QImage to ROS Image message and publish
        image = self.qimage_to_cv2(qimage)
        try:
            ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.publisher.publish(ros_image)
        except CvBridgeError as e:
            print(e)

    @staticmethod
    def qimage_to_cv2(qimage):
        # Convert QImage to OpenCV format
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        image_array = np.array(ptr).reshape(height, width, 3)  # Shape might need to be adjusted
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

class DrawingWindow(QMainWindow):
    def __init__(self, image_subscriber):
        super().__init__()
        self.image_subscriber = image_subscriber
        self.image_subscriber.update_image = self.display_image  # Link the update image method
        self.image_subscriber.update_right_wrist_images = self.display_right_wrist_images
        self.image_subscriber.update_left_wrist_images = self.display_left_wrist_images
        self.line_color = QColor(127, 0, 0)  # Default color
        self.default_color = QColor(127, 0, 0)  # Default color
        self.drawing_pixmap = QPixmap()  # Separate pixmap for drawing
        self.sketch_points = []  # List to store sketched waypoints
        self.selected_value = 127  # Default value
        self.DEFAULT_VALUE = 127
        self.annotate_mode = False
        self.contour_enabled = False  # Track if contour is enabled
        self.prediction = None
        self.direction_correction = None
        self.is_correction = False
        self.init_ui()
        self.image_publisher = ImagePublisher()  # Initialize your publisher
        self.robot_paused = False  # Track the pause state
        self.hl_paused = False
        self.use_preprogrammed_correction = False
        self.start_record = False
        self.pause_publisher = rospy.Publisher('/pause_robot', Bool, queue_size=10)
        self.pause_hl_publisher = rospy.Publisher('/pause_hl', Bool, queue_size=10)
        self.action_horizon_publisher = rospy.Publisher('/action_horizon', Int16, queue_size=10)
        self.direction_publisher = rospy.Publisher('/robot_direction', String, queue_size=10)  # New publisher for direction
        self.psm_pub = rospy.Publisher('/psm', String, queue_size=10)
        self.instructor_prediction_publisher = rospy.Publisher('/hl_policy_correction_phase_instruction', String, queue_size=10)
        self.instructor_prediction_pub = rospy.Publisher('/instructor_prediction', String, queue_size=10)
        self.correction_pub = rospy.Publisher('/direction_instruction_user', String, queue_size=1)
        self.start_record_pub = rospy.Publisher('/start_recording', Bool, queue_size=10)
        self.use_contour_pub = rospy.Publisher('/use_contour_image', Bool, queue_size=10)
        self.mid_level_action_request_pub = rospy.Publisher('/request_mid_level_action', Bool, queue_size=10)  # Publisher to request mid-level actions
        self.use_preprogrammed_correction_pub = rospy.Publisher('/use_preprogrammed_correction', Bool, queue_size=10)
        rospy.Subscriber('/mid_level_action', Float32MultiArray, self.mid_level_callback, queue_size=1)
        rospy.Subscriber('/instructor_prediction', String, self.instructor_prediction_callback, queue_size=1)
        rospy.Subscriber('/direction_instruction', String, self.direction_correction_callback, queue_size=1)
        rospy.Subscriber('/is_correction', Bool, self.is_correction_callback, queue_size=1)
        

    def init_ui(self):
                
        self.box_corners_grab = []
        self.box_corners_open = []
        
        self.drawing = False
        self.last_point = QPoint()
        self.image_label = QLabel(self)
        self.label_psm1 = QLabel(self)
        self.label_psm2 = QLabel(self)
        
        # Main layout setup
        main_layout = QHBoxLayout()
        image_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        # # Checkbox for toggling annotation mode
        # self.annotation_checkbox = QCheckBox("Annotate", self)
        # self.annotation_checkbox.stateChanged.connect(self.toggle_annotation_mode)
        # button_layout.addWidget(self.annotation_checkbox)
        
        # # Radio buttons for value selection
        # self.value_group = QButtonGroup(self)
        # radio_127 = QRadioButton("63")
        # radio_127.setChecked(True)
        # radio_0 = QRadioButton("0")
        # radio_255 = QRadioButton("190")
        # self.value_group.addButton(radio_127, 63)
        # self.value_group.addButton(radio_0, 0)
        # self.value_group.addButton(radio_255, 190)
        
        # # Connect radio buttons to update function
        # self.value_group.buttonClicked[int].connect(self.update_selected_value)

        # # Text field for custom value
        # self.custom_value_field = QLineEdit(self)
        # self.custom_value_field.setPlaceholderText('Enter custom value')
        # self.custom_value_field.returnPressed.connect(self.set_custom_value)

        # button_layout.addWidget(radio_127)
        # button_layout.addWidget(radio_0)
        # button_layout.addWidget(radio_255)
        # button_layout.addWidget(self.custom_value_field)

        # self.submit_button = QPushButton('Submit Sketch', self)
        # self.submit_button.clicked.connect(self.submit_sketch)
        # button_layout.addWidget(self.submit_button)

        # self.reset_button = QPushButton('Reset Sketch', self)
        # self.reset_button.clicked.connect(self.reset_sketch)
        # button_layout.addWidget(self.reset_button)
        
        # contour_button = QPushButton("segmentation", self)
        # contour_button.setStatusTip("use contour image")
        # contour_button.setCheckable(True)
        # contour_button.clicked.connect(self.publish_use_contour)
        # # contour_button.setIcon(QIcon(QPixmap("./icon/contour.png")))
        # button_layout.addWidget(contour_button)
        

        # speech_button = QPushButton(self)
        # speech_button.setStatusTip("speech to text")
        # speech_button.setCheckable(True)
        # speech_button.setIcon(QIcon(QPixmap("../icon/mic.png")))
		# # speech_button.setIconVisibleInMenu(True)
        # self.record_pressed = True
        # speech_button.clicked.connect(self.publish_speech_record)
        # button_layout.addWidget(speech_button)
        

        self.resize_field = QLineEdit(self)
        self.resize_field.setPlaceholderText('Enter action horizon')
        self.resize_field.returnPressed.connect(self.pub_action_horizon)
        button_layout.addWidget(self.resize_field)

        # self.avail_commands = ["grabbing gallbladder", "clipping first clip left tube", "going back first clip left tube", 
        #     "clipping second clip left tube", "going back second clip left tube",
        #     "clipping third clip left tube", "going back third clip left tube",
        #     "go to the cutting position left tube", "go back from the cut left tube",
        #     "clipping first clip right tube", "going back first clip right tube",
        #     "clipping second clip right tube", "going back second clip right tube",
        #     "clipping third clip right tube", "going back third clip right tube",
        #     "go to the cutting position right tube", "go back from the cut right tube",
        #     "needle pickup", "needle throw"
        #     ]
        
        self.avail_commands = ["needle pickup", "needle throw"]
        # self.avail_commands = ["1_needle_pickup", "2_needle_throw",  ]
        
        self.command_label = QLabel('Select Command:')
        self.command_combobox = QComboBox()
        self.command_combobox.addItems(self.avail_commands)
        button_layout.addWidget(self.command_combobox)
        ## add a button to send command according to currect selected command
        self.command_button = QPushButton('Send Command', self)
        self.command_button.clicked.connect(self.publish_instructor_prediction)
        button_layout.addWidget(self.command_button)
        ## add a button to start recording
        self.recording_button = QPushButton('Start Recording', self)
        self.recording_button.clicked.connect(self.start_recording_pub)
        button_layout.addWidget(self.recording_button)     
        
        # self.correction_commands = get_all_auto_labels_list()
        # self.correction_label = QLabel('Select Command:')
        # self.correction_combobox = QComboBox()
        # self.correction_combobox.addItems(self.correction_commands)
        # button_layout.addWidget(self.correction_combobox)
        # ## add a button to send command according to currect selected command
        # self.corretion_button = QPushButton('Send correction', self)
        # self.corretion_button.clicked.connect(self.publish_correction)
        
        # button_layout.addWidget(self.corretion_button)
        
        
        # # Add button to request mid-level actions
        # self.request_action_button = QPushButton('auto sketch', self)
        # self.request_action_button.clicked.connect(self.publish_request_mid_level_action)
        # button_layout.addWidget(self.request_action_button)

        # ## add a checkbox to use preprogrammed correction
        # self.use_preprogrammed_correction_checkbox = QCheckBox("preprogrammed correction", self)
        # self.use_preprogrammed_correction_checkbox.stateChanged.connect(self.use_preprogrammed_correction_callback)
        # button_layout.addWidget(self.use_preprogrammed_correction_checkbox)

        self.pause_button = QPushButton('Pause LL', self)
        self.pause_button.clicked.connect(self.toggle_robot_motion)
        button_layout.addWidget(self.pause_button)

        self.pause_hl_button = QPushButton('Pause HL', self)
        self.pause_hl_button.clicked.connect(self.toggle_hl_pause)
        button_layout.addWidget(self.pause_hl_button)
        # Connect the selection change event to the function
        # self.command_combobox.currentIndexChanged.connect(self.publish_instructor_prediction)
        
        # Add the QLabel widgets to the layout
        wrist_view_layout = QHBoxLayout()
        wrist_view_layout.addWidget(self.label_psm2)
        wrist_view_layout.addWidget(self.image_label)
        wrist_view_layout.addWidget(self.label_psm1)
        wrist_view_layout.setAlignment(self.label_psm2, Qt.AlignCenter)
        wrist_view_layout.setAlignment(self.image_label, Qt.AlignCenter)
        wrist_view_layout.setAlignment(self.label_psm1, Qt.AlignCenter)
        
        # Side panel for arrow buttons
        side_panel = QVBoxLayout()
        self.add_arrow_buttons(side_panel)
        # image_layout.addWidget(self.image_label)
        image_layout.addLayout(wrist_view_layout)
        image_layout.addLayout(button_layout)
        
        self.psm_label = QLabel('Select psm:')
        self.psm_combobox = QComboBox()
        self.psm_combobox.addItems(["psm1", "psm2"])
        self.psm_combobox.currentIndexChanged.connect(self.publish_psm)
        
        side_panel.addWidget(self.psm_combobox)
        # Add image label above the buttons
        # main_layout.addWidget(self.image_label)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(side_panel)

        
        # Set the central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setWindowTitle('Robot Learning Drawing Tool')
        self.show()

        
        
    def add_arrow_buttons(self, layout):
        # Add a frame to contain the arrow buttons
        arrow_frame = QFrame()
        arrow_layout = QVBoxLayout()

        # Create the up arrow button
        up_button = QPushButton("↑", self)
        up_button.clicked.connect(lambda: self.publish_direction("up"))
        arrow_layout.addWidget(up_button)

        # Create a layout for left and right buttons
        left_right_layout = QHBoxLayout()

        left_button = QPushButton("←", self)
        left_button.clicked.connect(lambda: self.publish_direction("left"))
        left_right_layout.addWidget(left_button)

        right_button = QPushButton("→", self)
        right_button.clicked.connect(lambda: self.publish_direction("right"))
        left_right_layout.addWidget(right_button)

        arrow_layout.addLayout(left_right_layout)

        # Create the down arrow button
        down_button = QPushButton("↓", self)
        down_button.clicked.connect(lambda: self.publish_direction("down"))
        arrow_layout.addWidget(down_button)

        # Add new buttons for Forward, Backward, Open, and Close
        forward_button = QPushButton("Forward", self)
        forward_button.clicked.connect(lambda: self.publish_direction("forward"))
        arrow_layout.addWidget(forward_button)

        backward_button = QPushButton("Backward", self)
        backward_button.clicked.connect(lambda: self.publish_direction("backward"))
        arrow_layout.addWidget(backward_button)

        open_button = QPushButton("Open", self)
        open_button.clicked.connect(lambda: self.publish_direction("open"))
        arrow_layout.addWidget(open_button)

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(lambda: self.publish_direction("close"))
        arrow_layout.addWidget(close_button)

        store_robot_pose_button = QPushButton("Store Pose", self)
        store_robot_pose_button.clicked.connect(lambda: self.publish_direction("store_pose"))
        arrow_layout.addWidget(store_robot_pose_button)
        
        move_to_pose_button = QPushButton("Move to Pose", self)
        move_to_pose_button.clicked.connect(lambda: self.publish_direction("move_to_pose"))
        arrow_layout.addWidget(move_to_pose_button)
        # Set the layout and add it to the provided layout
        arrow_frame.setLayout(arrow_layout)
        layout.addWidget(arrow_frame)


    ## ----------------- publishers -----------------
    def instructor_prediction_callback(self, msg):
        ## show the predictions on the GUI
        self.prediction = msg.data
    # def publish_speech_record(self):
    #     msg = Bool()
    #     if self.record_pressed:
    #         self.record_pressed = False
    #         msg.data = True    
    #         self.start_record_pub.publish(msg)
    #     else:
    #         self.record_pressed = True
    #         msg.data = False
    #         self.start_record_pub.publish(msg)
            
    def publish_use_contour(self):
        msg = Bool()
        self.contour_enabled = not self.contour_enabled  # Toggle the contour_enabled flag
        msg.data = self.contour_enabled
        self.use_contour_pub.publish(msg)
        
    def publish_direction(self, direction):
        self.direction_publisher.publish(direction)
    
    def publish_psm(self, psm):
        # Get the selected command
        selected_command = self.psm_combobox.currentText()
        
        # Publish the selected command to the /instructor_prediction topic
        self.psm_pub.publish(selected_command)
        print(f"Controlling {selected_command}")
                
    def publish_instructor_prediction(self, index):
        # Get the selected command
        selected_command = self.command_combobox.currentText()
        
        # Publish the selected command to the /instructor_prediction topic
        self.instructor_prediction_publisher.publish(selected_command)
        self.instructor_prediction_pub.publish(selected_command)
        
        print(f"Overwriting instruction prediction with {selected_command}")
    
    def start_recording_pub(self):
        self.start_record = not self.start_record  # Toggle pause state
        self.start_record_pub.publish(self.start_record)  # Publish the state
        print(f"Recording state: {self.start_record}")
        # Update button text based on state
        if self.start_record:
            self.recording_button.setText('Stop Recording')
        else:
            self.recording_button.setText('Start Recording')
    
    def publish_correction(self):
        selected_correction = self.correction_combobox.currentText()
        self.correction_pub.publish(selected_correction)
        self.direction_correction = selected_correction
        self.is_correction = True
        print(f"Overwriting direction with {selected_correction}")
    
    def publish_request_mid_level_action(self):
        self.reset_sketch() 
        msg = Bool()
        msg.data = True
        self.mid_level_action_request_pub.publish(msg)
        print("Requested mid-level action generation")
        
    def direction_correction_callback(self, msg):
        self.direction_correction = msg.data
        
    def is_correction_callback(self, msg):
        self.is_correction = msg.data
        
    def use_preprogrammed_correction_callback(self, state):
        print(state)
        self.use_preprogrammed_correction = True if state == 2 else False
        self.use_preprogrammed_correction_pub.publish(self.use_preprogrammed_correction)
        
    ## ----------------- callbacks -----------------
    def mid_level_callback(self, msg):
        # Update the action horizon field with the new value
        # action = np.array(msg.data).reshape(-1, 10)  # Assuming 10 values per action point
        
        # Step 1: Extract the data (flat list of 1000 elements)
        action_list = msg.data
        
        # Step 2: Convert to numpy array
        action_array = np.array(action_list, dtype=np.float32)
        
        # Step 3: Reshape the array back to 100x10
        reshaped_action = action_array.reshape((100, 10))
        
        # Now you can use reshaped_action as needed
        # print("Received action:", reshaped_action)
        # print("Received mid-level action")
        # self.reset_sketch()
        
        # self.draw_traj_on_pixmap(reshaped_action)
        # try:
        #     self.image_subscriber.image_updated()  # Update the image display
        # except:
        #     pass
        
    def draw_traj_on_pixmap(self, action, draw_depth_traj=True, draw_jaw_info=True):
        # square_thickness = 6
        # Example of drawing trajectory points on the drawing_pixmap
        painter = QPainter(self.drawing_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # For smoother edges
        # pen = QPen(QColor(255, 0, 0), 5, Qt.SolidLine)  # red pen for left trajectory
        # painter.setPen(pen)

        ee_l_points = (action[:, 0:2] + 1) / 2 * np.array([self.drawing_pixmap.width(), self.drawing_pixmap.height()])
        ee_r_points = (action[:, 5:7] + 1) / 2 * np.array([self.drawing_pixmap.width(), self.drawing_pixmap.height()])
        norm_depth_l = (action[:, 2] + 1) / 2 * 255
        norm_depth_r = (action[:, 7] + 1) / 2 * 255
        
        jaw_close_l_drawn = False
        jaw_close_r_drawn = False
        jaw_open_l_drawn = False
        jaw_open_r_drawn = False
        
        ee_l_jaw_close = action[:, 3]
        ee_l_jaw_open = action[:, 4]
        ee_r_jaw_close = action[:, 8]
        ee_r_jaw_open = action[:, 9]
        
        if ee_l_jaw_close[0] > 0.9 and ee_l_jaw_open[0] < 0.1:
            jaw_already_closed_l = True
        else:
            jaw_already_closed_l = False

        if ee_r_jaw_close[0] > 0.9 and ee_r_jaw_open[0] < 0.1:
            jaw_already_closed_r = True
        else:
            jaw_already_closed_r = False
            
            
        jaw_close_l = np.where(ee_l_jaw_close > 0.9)
        jaw_close_r = np.where(ee_r_jaw_close > 0.9)

        jaw_open_l = np.where(ee_l_jaw_open > 0.9)
        jaw_open_r = np.where(ee_r_jaw_open > 0.9)
        
        jaw_close_l = jaw_close_l[0]
        jaw_close_r = jaw_close_r[0]
        jaw_closing_index_l = jaw_close_l[0] if len(jaw_close_l) > 0 else None
        jaw_closing_index_r = jaw_close_r[0] if len(jaw_close_r) > 0 else None

        jaw_open_l = jaw_open_l[0]
        jaw_open_r = jaw_open_r[0]
        jaw_opening_index_l = jaw_open_l[0] if len(jaw_open_l) > 0 else None
        jaw_opening_index_r = jaw_open_r[0] if len(jaw_open_r) > 0 else None
        
        # print(jaw_closing_index_l, jaw_opening_index_l, jaw_closing_index_r, jaw_opening_index_r)

        for i in range(len(ee_l_points)):
            painter.setPen(QPen(QColor(int(norm_depth_l[i]), 0, 0), LINEWIDTH, Qt.SolidLine))  # Red pen for right trajectory
            
            painter.drawEllipse(QPointF(ee_l_points[i][0], ee_l_points[i][1]), RADIUS, RADIUS)
            painter.setPen(QPen(QColor(int(norm_depth_r[i]), 0, 0), LINEWIDTH, Qt.SolidLine))  # Red pen for right trajectory
            painter.drawEllipse(QPointF(ee_r_points[i][0], ee_r_points[i][1]), RADIUS, RADIUS)
            
            if jaw_closing_index_l is not None and not jaw_already_closed_l:
                if i == jaw_closing_index_l and not jaw_close_l_drawn:
                    pos = (int(ee_l_points[i][0]), int(ee_l_points[i][1]))
                    ## draw a green square to indicate jaw closing
                    painter.setPen(QPen(Qt.green, LINEWIDTH, Qt.SolidLine))  # Green pen for the box
                    painter.drawRect(QRect(pos[0] - 15, pos[1] - 15, 30, 30))  # Draw the box
                    
                    jaw_close_l_drawn = True
                                    
            if jaw_opening_index_l is not None and jaw_already_closed_l:
                if i == jaw_opening_index_l and not jaw_open_l_drawn:
                    pos = (int(ee_l_points[i][0]), int(ee_l_points[i][1]))
                    ## draw a blue square to indicate jaw opening
                    painter.setPen(QPen(Qt.blue, LINEWIDTH, Qt.SolidLine))
                    painter.drawRect(QRect(pos[0] - 15, pos[1] - 15, 30, 30))  # Draw the box
                    
            if jaw_closing_index_r is not None and not jaw_already_closed_r:
                # print("drawing jaw closing")
                if i == jaw_closing_index_r and not jaw_close_r_drawn:
                    pos = (int(ee_r_points[i][0]), int(ee_r_points[i][1]))
                    # print("right jaw closing idx", i)
                    ## draw a green square to indicate jaw closing
                    painter.setPen(QPen(Qt.green, LINEWIDTH, Qt.SolidLine))
                    painter.drawRect(QRect(pos[0] - 15, pos[1] - 15, 30, 30))  # Draw the box
            
            if jaw_opening_index_r is not None and jaw_already_closed_r:
                if i == jaw_opening_index_r and not jaw_open_r_drawn:
                    pos = (int(ee_r_points[i][0]), int(ee_r_points[i][1]))
                    ## draw a blue square to indicate jaw opening
                    painter.setPen(QPen(Qt.blue, LINEWIDTH, Qt.SolidLine))
                    painter.drawRect(QRect(pos[0] - 15, pos[1] - 15, 30, 30))  # Draw the box
                
        painter.end()
        
        
    def toggle_annotation_mode(self, state):
        self.annotate_mode = bool(state)
        self.annotation_checkbox.setChecked(self.annotate_mode)
        
    def update_selected_value(self, value):
        self.selected_value = value
        
    def set_custom_value(self):
        value = self.custom_value_field.text()
        if value.isdigit() and 0 <= int(value) <= 255:
            self.selected_value = int(value)
            self.custom_value_field.clear()
                
    def toggle_robot_motion(self):
        self.robot_paused = not self.robot_paused  # Toggle pause state
        self.pause_publisher.publish(self.robot_paused)  # Publish the state

        # Update button text based on state
        if self.robot_paused:
            self.pause_button.setText('Resume LL')
        else:
            self.pause_button.setText('Pause LL')
        
    def toggle_hl_pause(self):
        self.hl_paused = not self.hl_paused  # Toggle pause state
        self.pause_hl_publisher.publish(self.hl_paused)  # Publish the state

        # Update button text based on state
        if self.hl_paused:
            self.pause_hl_button.setText('Resume HL')
        else:
            self.pause_hl_button.setText('Pause HL')
            
    def display_left_wrist_images(self, image_psm2):
        if image_psm2 is not None:
            height, width, channel = image_psm2.shape
            bytes_per_line = 3 * width
            q_img_psm2 = QImage(image_psm2.data, width, height, bytes_per_line, QImage.Format_RGB888)
            image_pixmap_psm2 = QPixmap.fromImage(q_img_psm2)
            self.label_psm2.setPixmap(image_pixmap_psm2)
            
    def display_right_wrist_images(self, image_psm1):
        if image_psm1 is not None:
            height, width, channel = image_psm1.shape
            bytes_per_line = 3 * width
            q_img_psm1 = QImage(image_psm1.data, width, height, bytes_per_line, QImage.Format_RGB888)
            image_pixmap_psm1 = QPixmap.fromImage(q_img_psm1)
            self.label_psm1.setPixmap(image_pixmap_psm1)

            
    def display_image(self, image):
        """ Convert the OpenCV image to QPixmap and display it. """
        # print(image.shape)
        
        height, width, channel = image.shape
        if self.prediction:
            command = self.prediction
            # if self.direction_correction:
            if self.is_correction and self.direction_correction and self.direction_correction != "do not move":
                command = f" {self.direction_correction}"
            ## use cv2 to write self.prediction on the image
            cv2.putText(image, command, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        image_pixmap = QPixmap.fromImage(q_img)
        if self.drawing_pixmap.isNull():  # Initialize drawing pixmap size the first time
            self.drawing_pixmap = QPixmap(image_pixmap.size())
            self.drawing_pixmap.fill(Qt.transparent)
        self.merge_pimas(image_pixmap, self.drawing_pixmap)

    def merge_pimas(self, background, overlay):
        box_thickness = 6
        # Updated merge method to handle highlights and values
        if not background or background.isNull():
            return
        temp_pixmap = QPixmap(background.size())
        temp_pixmap.fill(Qt.transparent)
        painter = QPainter(temp_pixmap)
        painter.drawPixmap(0, 0, background)
        for point in self.sketch_points:
            if point.get('highlight', False):
                color_intensity = point['value']
                # Ensure the color intensity is within the valid range of 0-255
                color_intensity = max(0, min(255, color_intensity))
                pen = QPen(QColor(color_intensity, 0, 0), LINEWIDTH, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawPoint(point['pos'])
                painter.drawText(point['pos'] + QPoint(10, -10), str(point['value']))
                # Set the brush to fill the circle with the same color as the pen
                brush = QBrush(QColor(color_intensity, 0, 0))
                painter.setBrush(brush)
                
                # Calculate position and radius
                center = point['pos']

                # Draw the circle
                painter.drawEllipse(center, RADIUS, RADIUS)
            else:
                color_intensity = int(point['value'])
                
                painter.setPen(QPen(QColor(color_intensity, 0, 0), LINEWIDTH, Qt.SolidLine))
                painter.drawPoint(point['pos'])
                # Set the brush to fill the circle with the same color as the pen
                brush = QBrush(QColor(color_intensity, 0, 0))
                painter.setBrush(brush)
                
                # Calculate position and radius
                center = point['pos']

                # Draw the circle
                painter.drawEllipse(center, RADIUS, RADIUS)
        
        brush = QBrush(Qt.transparent)
        painter.setBrush(brush)
        
        if self.box_corners_grab:
            painter.setPen(QPen(Qt.green, box_thickness, Qt.SolidLine))  # Green pen for the box
            for start, end in self.box_corners_grab:
                rect = QRect(start, end)
                painter.drawRect(rect)  # Draw the box
        elif self.box_corners_open:
            painter.setPen(QPen(Qt.blue, box_thickness, Qt.SolidLine))  # Green pen for the box
            for start, end in self.box_corners_open:
                rect = QRect(start, end)
                painter.drawRect(rect)  # Draw the box   
                  
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        self.image_label.setPixmap(temp_pixmap)

    def pub_action_horizon(self):
        horizon = self.resize_field.text()
        if horizon.isdigit():
            self.action_horizon_publisher.publish(int(horizon))
            # Clear the text field after publishing
            self.resize_field.clear()
        else:
            print("Invalid action horizon")
            
    def mousePressEvent(self, event):
        # Get the position of the image label
        x_img, y_img, w_img, h_img = self.image_label.geometry().getRect()

        # Calculate the mouse click position relative to the image label
        x_rel = event.x() - x_img
        y_rel = event.y() - y_img

        # Check if the click was inside the image label
        if 0 <= x_rel <= w_img and 0 <= y_rel <= h_img:
            # The click was inside the image label, use the relative coordinates
            click_pos = QPoint(x_rel, y_rel)
        else:
            # The click was outside the image label, ignore it
            return

        box_size = 15
        if event.button() == Qt.LeftButton:
            self.drawing = True
            if self.annotate_mode:
                self.annotate_depth(click_pos)
            else:
                # Start sketching a new line if annotation mode is off
                self.last_point = click_pos
                self.sketch_points.append({'pos': self.last_point, 'value': self.DEFAULT_VALUE, 'highlight': False})
                self.merge_pimas(self.image_label.pixmap(), self.drawing_pixmap)
                
        if event.button() == Qt.MidButton:
            start_point = click_pos
            top_left = QPoint(start_point.x() - box_size, start_point.y() - box_size)
            bottom_Right = QPoint(start_point.x() + box_size, start_point.y() + box_size)

            # Append the start point and a dummy end point (same as start initially)
            self.box_corners_grab.append([top_left, bottom_Right])
            self.merge_pimas(self.image_label.pixmap(), self.drawing_pixmap)

        if event.button() == Qt.XButton1:
            start_point = click_pos
            top_left = QPoint(start_point.x() - box_size, start_point.y() - box_size)
            bottom_Right = QPoint(start_point.x() + box_size, start_point.y() + box_size)

            # Append the start point and a dummy end point (same as start initially)
            self.box_corners_open.append([top_left, bottom_Right])
            self.merge_pimas(self.image_label.pixmap(), self.drawing_pixmap)

    def annotate_depth(self, click_pos):
        # Find the closest existing sketch point
        closest_point = None
        min_distance = float('inf')
        for point in self.sketch_points:
            dist = (point['pos'].x() - click_pos.x())**2 + (point['pos'].y() - click_pos.y())**2
            if dist < min_distance:
                min_distance = dist
                closest_point = point
        if closest_point:
            # Update the closest point with the selected value and highlight it
            closest_point['value'] = self.selected_value
            closest_point['highlight'] = True
            self.global_interpolation()
    
    def global_interpolation(self):
        # Find all annotated points
        annotated_indices = [i for i, point in enumerate(self.sketch_points) if point['highlight']]
        annotated_indices.insert(0, 0)  # Add the first point to the list
        if not annotated_indices:
            return
        # Interpolate between all successive pairs of annotated points
        for i in range(len(annotated_indices) - 1):
            start_index = annotated_indices[i]
            end_index = annotated_indices[i + 1]
            start_value = self.sketch_points[start_index]['value']
            end_value = self.sketch_points[end_index]['value']
            num_points = end_index - start_index
            for j in range(1, num_points):
                interpolated_value = start_value + (end_value - start_value) * j / num_points
                self.sketch_points[start_index + j]['value'] = interpolated_value

        # Optionally, redraw the sketch points
        # self.redraw_sketch_points()

    def keyPressEvent(self, event):
        # Check if 'e' is pressed
        if event.key() == Qt.Key_E:
            self.annotate_mode = not self.annotate_mode  # Toggle the annotate variable
            print(f"Annotate toggled to {self.annotate_mode}")  # Optional: print the toggle state to the console
            self.toggle_annotation_mode(self.annotate_mode)
            self.update()  # Redraw the widget if needed
        
        # if event.key() == Qt.Key_R:
        #     self.reset_sketch()  # Reset the sketch points
        #     self.toggle_annotation_mode(False)
        #     self.update()  # Redraw the widget if needed
            
        if event.key() == Qt.Key_P:
            self.toggle_robot_motion() 
        
    def mouseMoveEvent(self, event):
        # Get the position of the image label
        x_img, y_img, w_img, h_img = self.image_label.geometry().getRect()

        # Calculate the mouse move position relative to the image label
        x_rel = event.x() - x_img
        y_rel = event.y() - y_img

        # Check if the move was inside the image label
        if 0 <= x_rel <= w_img and 0 <= y_rel <= h_img:
            # The move was inside the image label, use the relative coordinates
            move_pos = QPoint(x_rel, y_rel)
        else:
            # The move was outside the image label, ignore it
            return

        if event.buttons() == Qt.LeftButton and self.drawing:
            self.last_point = move_pos
            self.sketch_points.append({'pos': self.last_point, 'value': self.DEFAULT_VALUE, 'highlight': False})
            self.merge_pimas(self.image_label.pixmap(), self.drawing_pixmap)

    def redraw_sketch_points(self):
        # Clear the drawing pixmap to redraw all points
        self.drawing_pixmap.fill(Qt.transparent)

        painter = QPainter(self.drawing_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # For smoother edges

        for point in self.sketch_points:
            color_intensity = int(point['value'])
            print(color_intensity)
            # Ensure the color intensity is within the valid range of 0-255
            color_intensity = max(0, min(255, color_intensity))
            pen = QPen(QColor(color_intensity, 0, 0), 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPoint(point['pos'])
            # Set the brush to fill the circle with the same color as the pen
            brush = QBrush(QColor(color_intensity, 0, 0))
            painter.setBrush(brush)
            
            # Calculate position and radius
            center = point['pos']

            # Draw the circle
            painter.drawEllipse(center, RADIUS, RADIUS)

        painter.end()
        
    def merge_pixmaps(self, background, overlay):
        """ This function merges the background pixmap with an overlay pixmap """
        if not background or background.isNull():
            return
        temp_pixmap = QPixmap(background.size())
        temp_pixmap.fill(Qt.transparent)
        painter = QPainter(temp_pixmap)
        painter.drawPixmap(0, 0, background)
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        self.image_label.setPixmap(temp_pixmap)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def change_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.line_color = color

    def submit_sketch(self):
        # Get the QPixmap from the label, convert it to QImage for publishing
        pixmap = self.image_label.pixmap()
        if pixmap:
            qimage = pixmap.toImage()
            self.image_publisher.publish_image(qimage)
            print(self.sketch_points)
            # self.sketch_points = [] # clear sketch points
        else:
            print("No image to submit.")

    def reset_sketch(self):
        self.drawing_pixmap.fill(Qt.transparent)
        self.sketch_points = []  # Clear the list of points
        self.box_corners_open = []
        self.box_corners_grab = []
        self.merge_pimas(self.image_label.pixmap(), self.drawing_pixmap)

    def resize_image(self):
        size = self.resize_field.text().split(',')
        if len(size) == 2 and size[0].isdigit() and size[1].isdigit():
            new_width = int(size[0])
            new_height = int(size[1])
            self.drawing_pixmap = self.drawing_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
            self.merge_pimas(self.image_label.pixmap(), self.drawing_pixmap)
            self.resize_field.clear()

def main():
    rospy.init_node('image_subscriber_node', anonymous=True)
    app = QApplication(sys.argv)
    ex = DrawingWindow(ImageSubscriber())
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()