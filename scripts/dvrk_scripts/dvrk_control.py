from rostopics import ros_topics
import rospy
import time

import argparse
import sys
import time
import crtk
import dvrk
import math
import numpy as np
import PyKDL

# rt = ros_topics() 
# rospy.init_node('rostopic_recorder', anonymous=True)
# time.sleep(0.5)
# print(rt.psm1_pose)

# example of application using arm.py
class example_application:

    # configuration
    def __init__(self, ral, arm_name, expected_interval):
        print('configuring dvrk_arm_test for {}'.format(arm_name))
        self.ral = ral
        self.expected_interval = expected_interval
        self.arm = dvrk.psm(ral = ral,
                            arm_name = arm_name,
                            expected_interval = expected_interval)
        self.arm_name = arm_name
        # self.arm = dvrk.psm(arm_name = arm_name,
        #             expected_interval = expected_interval)

    # homing example
    def home(self):
        self.arm.check_connections()

        print('starting enable')
        if not self.arm.enable(10):
            sys.exit('failed to enable within 10 seconds')
        print('starting home')
        if not self.arm.home(10):
            sys.exit('failed to home within 10 seconds')
        # get current joints just to set size
        print('move to starting position')
        goal = numpy.copy(self.arm.setpoint_jp())
        # go to zero position, for PSM and ECM make sure 3rd joint is past cannula
        goal.fill(0)
        if ((self.arm.name() == 'PSM1') or (self.arm.name() == 'PSM2')
            or (self.arm.name() == 'PSM3') or (self.arm.name() == 'ECM')):
            goal[2] = 0.12
        # move and wait
        print('moving to starting position')
        self.arm.move_jp(goal).wait()
        # try to move again to make sure waiting is working fine, i.e. not blocking
        print('testing move to current position')
        move_handle = self.arm.move_jp(goal)
        print('move handle should return immediately')
        move_handle.wait()
        print('home complete')

    # get methods
    def run_get(self):
        [p, v, e, t] = self.arm.measured_js()
        d = self.arm.measured_jp()
        [d, t] = self.arm.measured_jp(extra = True)
        d = self.arm.measured_jv()
        [d, t] = self.arm.measured_jv(extra = True)
        d = self.arm.measured_jf()
        [d, t] = self.arm.measured_jf(extra = True)
        d = self.arm.measured_cp()
        [d, t] = self.arm.measured_cp(extra = True)
        d = self.arm.local.measured_cp()
        [d, t] = self.arm.local.measured_cp(extra = True)
        d = self.arm.measured_cv()
        [d, t] = self.arm.measured_cv(extra = True)
        d = self.arm.body.measured_cf()
        [d, t] = self.arm.body.measured_cf(extra = True)
        d = self.arm.spatial.measured_cf()
        [d, t] = self.arm.spatial.measured_cf(extra = True)

        [p, v, e, t] = self.arm.setpoint_js()
        d = self.arm.setpoint_jp()
        [d, t] = self.arm.setpoint_jp(extra = True)
        d = self.arm.setpoint_jv()
        [d, t] = self.arm.setpoint_jv(extra = True)
        d = self.arm.setpoint_jf()
        [d, t] = self.arm.setpoint_jf(extra = True)
        d = self.arm.setpoint_cp()
        [d, t] = self.arm.setpoint_cp(extra = True)
        d = self.arm.local.setpoint_cp()
        [d, t] = self.arm.local.setpoint_cp(extra = True)

    # direct joint control example
    def run_servo_jp(self):
        print('starting servo_jp')
        # get current position
        initial_joint_position = numpy.copy(self.arm.setpoint_jp())
        print('testing direct joint position for 2 joints out of %i' % initial_joint_position.size)
        amplitude = math.radians(5.0) # +/- 5 degrees
        duration = 5  # seconds
        samples = duration / self.expected_interval
        # create a new goal starting with current position
        goal_p = numpy.copy(initial_joint_position)
        goal_v = numpy.zeros(goal_p.size)
        start = time.time()

        sleep_rate = self.ral.create_rate(1.0 / self.expected_interval)
        for i in range(int(samples)):
            angle = i * math.radians(360.0) / samples
            goal_p[0] = initial_joint_position[0] + amplitude * (1.0 - math.cos(angle))
            goal_p[1] = initial_joint_position[1] + amplitude *  (1.0 - math.cos(angle))
            goal_v[0] = amplitude * math.sin(angle)
            goal_v[1] = goal_v[0]
            self.arm.servo_jp(goal_p, goal_v)
            sleep_rate.sleep()

        actual_duration = time.time() - start
        print('servo_jp complete in %2.2f seconds (expected %2.2f)' % (actual_duration, duration))

    # goal joint control example
    def run_move_jp(self):
        print('starting move_jp')
        # get current position
        initial_joint_position = numpy.copy(self.arm.setpoint_jp())
        print('testing goal joint position for 2 joints out of %i' % initial_joint_position.size)
        amplitude = math.radians(10.0)
        # create a new goal starting with current position
        goal = numpy.copy(initial_joint_position)
        # first motion
        goal[0] = initial_joint_position[0] + amplitude
        goal[1] = initial_joint_position[1] - amplitude
        self.arm.move_jp(goal).wait()
        # second motion
        goal[0] = initial_joint_position[0] - amplitude
        goal[1] = initial_joint_position[1] + amplitude
        self.arm.move_jp(goal).wait()
        # back to initial position
        self.arm.move_jp(initial_joint_position).wait()
        print('move_jp complete')

    # utility to position tool/camera deep enough before cartesian examples
    def prepare_cartesian(self):
        # make sure the camera is past the cannula and tool vertical
        goal = numpy.copy(self.arm.setpoint_jp())
        if ((self.arm.name().endswith('PSM1')) or (self.arm.name().endswith('PSM2'))
            or (self.arm.name().endswith('PSM3')) or (self.arm.name().endswith('ECM'))):
            print('preparing for cartesian motion')
            # set in position joint mode
            goal[0] = 0.0
            goal[1] = 0.0
            goal[2] = 0.12
            goal[3] = 0.0
            self.arm.move_jp(goal).wait()

    # direct cartesian control example
    def run_servo_cp(self):
        print('starting servo_cp')
        self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M
        # motion parameters
        amplitude = 0.02 # 4 cm total
        duration = 5  # 5 seconds
        samples = duration / self.expected_interval
        start = time.time()

        sleep_rate = self.ral.create_rate(1.0 / self.expected_interval)
        for i in range(int(samples)):
            goal.p[0] =  initial_cartesian_position.p[0] + amplitude *  (1.0 - math.cos(i * math.radians(360.0) / samples))
            goal.p[1] =  initial_cartesian_position.p[1] + amplitude *  (1.0 - math.cos(i * math.radians(360.0) / samples))
            self.arm.servo_cp(goal)
            # check error on kinematics, compare to desired on arm.
            # to test tracking error we would compare to
            # current_position
            setpoint_cp = self.arm.setpoint_cp()
            errorX = goal.p[0] - setpoint_cp.p[0]
            errorY = goal.p[1] - setpoint_cp.p[1]
            errorZ = goal.p[2] - setpoint_cp.p[2]
            error = math.sqrt(errorX * errorX + errorY * errorY + errorZ * errorZ)
            if error > 0.002: # 2 mm
                print('Inverse kinematic error in position [%i]: %s (might be due to latency)' % (i, error))
            sleep_rate.sleep()

        actual_duration = time.time() - start
        print('servo_cp complete in %2.2f seconds (expected %2.2f)' % (actual_duration, duration))

    # direct cartesian control example
    def run_move_cp(self):
        print('starting move_cp')
        # self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M

        # motion parameters
        amplitude = 0.001 # 5 cm

        # first motion
        goal.p[0] =  initial_cartesian_position.p[0] - amplitude
        goal.p[1] =  initial_cartesian_position.p[1]
        self.arm.move_cp(goal).wait()
        # second motion
        goal.p[0] =  initial_cartesian_position.p[0] + amplitude
        goal.p[1] =  initial_cartesian_position.p[1]
        self.arm.move_cp(goal).wait()
        # back to initial position
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1]
        self.arm.move_cp(goal).wait()
        # first motion
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1] - amplitude
        self.arm.move_cp(goal).wait()
        # second motion
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1] + amplitude
        self.arm.move_cp(goal).wait()
        # back to initial position
        goal.p[0] =  initial_cartesian_position.p[0]
        goal.p[1] =  initial_cartesian_position.p[1]
        self.arm.move_cp(goal).wait()
        print('move_cp complete')

    def run_full_pose_goal(self, wp):
        # print('starting move_cp_goal')
        # self.prepare_cartesian()
        # create a new goal starting with current position

        pos_diff_thres = 0.005 # 1 cm
        rot_diff_thres = 0.02 # radians
        jaw_diff_thres = 0.1 # radians
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        jaw_position = self.arm.jaw.setpoint_jp()
        # print("jaw init", initial_jaw_position)
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M

        goal.p[0] =  wp[0]
        goal.p[1] =  wp[1]
        goal.p[2] =  wp[2]
        goal.M = PyKDL.Rotation.Quaternion(wp[3], wp[4], wp[5], wp[6])

        ## calculate difference 
        init_arr = np.array((initial_cartesian_position.p.x(),
                             initial_cartesian_position.p.y(),
                             initial_cartesian_position.p.z()))
        diff = wp[0:3] - init_arr
        diff_norm = np.linalg.norm(diff)
        
        ### calculate orientation difference
        ## use GetRotAngle to get rot angle difference
        rot_diff = initial_cartesian_position.M.Inverse() * goal.M
        angle_diff, _ = rot_diff.GetRotAngle()
        
        jaw_diff = wp[-1] - jaw_position[0]

        ## use self.arm.servo_cp and self.jaw.servo_jp
        ## if difference too large -> use move_cp to prevent tracking error for servo_cp
        # if abs(diff_norm) > pos_diff_thres or abs(angle_diff) > rot_diff_thres:
        # #     print(diff_norm, angle_diff)
        #     # print("move cp")
        #     self.arm.move_cp(goal).wait()
        # else:
        #     print("servo cp")
        # self.arm.move_cp(goal).wait()
        
        self.arm.servo_cp(goal)
            
        print("move cp complete")
        
        # if abs(jaw_diff) > jaw_diff_thres:
        #     print("open jaw")
        #     self.arm.jaw.open(angle = wp[-1]).wait() # reach gripper angle
        
        # else:
        print("servo jaw")
        
        jaw_position[0] = wp[-1]
        self.arm.jaw.servo_jp(jaw_position)
        print("jaw open complete")
        

    def move_robot_in_direction(self, direction):
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        initial_jaw_position = self.arm.jaw.setpoint_jp()

        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M
        
        translation = ['up', 'down', 'left', 'right', 'forward', 'backward', 'higher', 'lower', 'to the left', 'to the right', 'away from me', 'towards me']
        jaw_motion = ['open', 'close']
        if direction in translation:
            if direction == 'up' or direction == "higher":
                goal.p = initial_cartesian_position.p
                goal.p[1] += 0.001
                
            if direction == 'down' or direction == "lower":
                goal.p = initial_cartesian_position.p
                goal.p[1] -= 0.001
                
            if direction == 'left' or direction == "to the left":
                goal.p = initial_cartesian_position.p
                goal.p[0] += 0.001
                
            if direction == 'right' or direction == "to the right":
                print("I am here")
                goal.p = initial_cartesian_position.p
                goal.p[0] -= 0.001
                
            if direction == 'forward' or direction == "away from me":
                goal.p = initial_cartesian_position.p
                goal.p[2] += 0.001
                
            if direction == 'backward' or direction == "towards me":
                goal.p = initial_cartesian_position.p
                goal.p[2] -= 0.001
                
            self.arm.servo_cp(goal)
            self.arm.jaw.servo_jp(initial_jaw_position)
            
        elif direction in jaw_motion:
            if direction == 'open':
                self.arm.servo_cp(initial_cartesian_position)
                
                self.arm.jaw.open(angle = 0.6).wait()
                
            if direction == 'close':
                self.arm.servo_cp(initial_cartesian_position)
                
                self.arm.jaw.open(angle = -0.5).wait()
        
    def move_robot_to_pose(self, pose):
        self.arm.move_cp(pose).wait()
        
    def run_full_pose_goal_bimanual(self, wp):
        # print('starting move_cp_goal')
        # self.prepare_cartesian()

        # create a new goal starting with current position
        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.arm.setpoint_cp().p
        initial_cartesian_position.M = self.arm.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = self.arm.setpoint_cp().M

        goal.p[0] =  wp[0]
        goal.p[1] =  wp[1]
        goal.p[2] =  wp[2]
        goal.M = PyKDL.Rotation.Quaternion(wp[3], wp[4], wp[5], wp[6])

        # first motion
        # print("initial cartesian position", initial_cartesian_position.p)
        # print("goal", goal.p)
        # print("diff in vector space", goal.p - initial_cartesian_position.p)

        init_arr = np.array((initial_cartesian_position.p.x(),
                             initial_cartesian_position.p.y(),
                             initial_cartesian_position.p.z()))
        diff = wp[0:3] - init_arr
        diff_norm = np.linalg.norm(diff)
        diff = diff * 1000
        diff_norm = diff_norm * 1000
        # print("diff:" , diff, "mm")
        # print("diff_norm:", diff_norm, "mm")
        # input("Press enter to execute....")
        self.arm.move_cp(goal).wait()
        # time.sleep(0.1)
        self.arm.jaw.open(angle = wp[-1]).wait() # reach gripper angle
        # print("desired_angle", wp[-1])
        # time.sleep(0.1)
        # print('move_cp complete')
        

    def move_cp_rot(self, goal_r):
        print('starting move_cp_goal')
        # self.prepare_cartesian()

        # create a new goal starting with current position
        goal = PyKDL.Frame()
        goal.p = self.arm.setpoint_cp().p
        goal.M = PyKDL.Rotation.Quaternion(goal_r[0], goal_r[1], goal_r[2], goal_r[3])
        print(goal.M)

        # first motion
        self.arm.move_cp(goal).wait()

    def run_jaw_servo(self):
        print('starting jaw servo')
        # try to open and close directly, needs interpolation
        print('close and open without other servo command')
        input("    Press Enter to continue...")
        start_angle = math.radians(-0.6)
        # start_angle = -0.6
        self.arm.jaw.open(angle = start_angle).wait()
        # assume we start at 30 the move +/- 30
        # amplitude = math.radians(30.0)
        # duration = 5  # seconds
        # samples = int(duration / self.expected_interval)
        # # create a new goal starting with current position
        # for i in range(samples * 4):
        #     goal = start_angle + amplitude * (math.cos(i * math.radians(360.0) / samples) - 1.0)
        #     self.arm.jaw.servo_jp(np.array(goal))
        #     rospy.sleep(self.expected_interval)



    # main method
    def run(self, goal):
        # self.home()
        # self.run_get()
        # self.run_servo_jp()
        # self.run_move_jp()
        # self.run_servo_cp()
        # self.run_move_cp()
        self.run_move_cp_goal(goal)

if __name__ == '__main__':
    # extract ros arguments (e.g. __ns:= for namespace)
    argv = crtk.ral.parse_argv(sys.argv[1:]) # skip argv[0], script name

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'],
                        help = 'arm name corresponding to ROS topics without namespace.  Use __ns:= to specify the namespace')
    parser.add_argument('-i', '--interval', type=float, default=0.01,
                        help = 'expected interval in seconds between messages sent by the device')
    args = parser.parse_args(argv)

    # change me as needed
    goal_rot = np.array((0.83877435452268, 0.343168783321817, 
                         -0.385516485863504, -0.173406480490033))

    ral = crtk.ral('dvrk_arm_test')
    application = example_application(ral, args.arm, args.interval)
    ral.spin_and_execute(application.run_jaw_servo)
    # ral.spin_and_execute(application.run)
