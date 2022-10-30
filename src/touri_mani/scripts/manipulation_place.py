#!/usr/bin/env python

from __future__ import print_function

import math
import time
import threading
import sys
import tf2_ros
import argparse as ap        
import numpy as np
import os
import rospy
import actionlib
from sensor_msgs.msg import PointCloud2, JointState
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from std_msgs.msg import String, Bool
import hello_helpers.hello_misc as hm

# -----------------------------------------------------------------------------

class GraspObjectNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        # self.move_base = nv.MoveBase(self)
        self.letter_height_m = 0.2
        self.wrist_position = None
        self.lift_position = None
        self.manipulation_view = None
        self.debug_directory = None
        self.defined = False
        
    def joint_states_callback(self, joint_states):
        with self.joint_states_lock: 
            self.joint_states = joint_states
        wrist_position, wrist_velocity, wrist_effort = hm.get_wrist_state(joint_states)
        self.wrist_position = wrist_position
        lift_position, lift_velocity, lift_effort = hm.get_lift_state(joint_states)
        self.lift_position = lift_position
        self.left_finger_position, temp1, temp2 = hm.get_left_finger_state(joint_states)
        
    def lower_tool_until_contact(self):
        rospy.loginfo('lower_tool_until_contact')
        trigger_request = TriggerRequest() 
        trigger_result = self.trigger_lower_until_contact_service(trigger_request)
        rospy.loginfo('trigger_result = {0}'.format(trigger_result))
        
    def move_to_initial_configuration(self):
        initial_pose = {'wrist_extension': 0.01,
                        'joint_wrist_yaw': 0.0,
                        'gripper_aperture': 0.125}

        rospy.loginfo('Move to the initial configuration for drawer opening.')
        self.move_to_pose(initial_pose)

    def look_at_surface(self, scan_time_s=None):
        self.manipulation_view = mp.ManipulationView(self.tf2_buffer, self.debug_directory)
        manip = self.manipulation_view
        head_settle_time_s = 0.02 #1.0
        manip.move_head(self.move_to_pose)
        rospy.sleep(head_settle_time_s)
        if scan_time_s is None:
            manip.update(self.point_cloud, self.tf2_buffer)
        else:
            start_time_s = time.time()
            while ((time.time() - start_time_s) < scan_time_s): 
                manip.update(self.point_cloud, self.tf2_buffer)
        if self.debug_directory is not None:
            dirname = self.debug_directory + 'grasp_object/'
            # If the directory does not already exist, create it.
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = 'look_at_surface_' + hm.create_time_string()
            manip.save_scan(dirname + filename)
        else:
            rospy.loginfo('GraspObjectNode: No debug directory provided, so debugging data will not be saved.')

    def drive(self, forward_m):
        tolerance_distance_m = 0.005
        if forward_m > 0: 
            at_goal = self.move_base.forward(forward_m, detect_obstacles=False, tolerance_distance_m=tolerance_distance_m)
        else:
            at_goal = self.move_base.backward(forward_m, detect_obstacles=False, tolerance_distance_m=tolerance_distance_m)
        
    def trigger_place_object_callback(self, request):
        rospy.loginfo("place service called")
        # pose = {'joint_head_tilt':(np.deg2rad(-60))}
        # self.move_to_pose(pose)


        # Step 0 : Moving the camera to detect object 
        # i = 1
        # while self.detected.data!=True:
        #     rospy.loginfo("Value:"+str(self.detected.data))
        #     rospy.loginfo("Rotating clockwise : "+str(i*20)+"by 20 degrees")
        #     # rot_angle = np.deg2rad(-20) * i
        #     pose = {'joint_head_pan': np.deg2rad(-20) * i}
        #     i=i+1
        #     if i>9:
        #         rospy.loginfo("Rotating clockwise : "+str((i-9)*20)+"by 10 degrees")
        #         pose = {'joint_head_pan': np.deg2rad(20) * (i-9)}
        #     rospy.loginfo(pose)
        #     self.move_to_pose(pose)
        #     if(i>18):
        #         rospy.loginfo("Couldn't detect object")
        #         return
        #     rospy.loginfo("Done moving")
        #     time.sleep(2)

        open = 0.15283721447
        close=-0.37
        
        # Steps
        if self.goal_x and self.goal_y and self.goal_z:
            
            poses = [
                # Step 0 - Home position
                # {'joint_lift': 0.6}
                {'joint_lift': 0.9,'wrist_extension': 0.0},
                # Step 1 - Rotate to bring the cup in center
                {'translate_mobile_base': self.goal_x - 0.65},
                # Step 2
                {'rotate_mobile_base': 1.57},
                # Step 3
                {'wrist_extension': 0.4},
                # Step 4
                # {'joint_lift': self.goal_z + 0.07},
                {'joint_lift': 0.76},
                # Step 5
                #{'translate_mobile_base': self.goal_x},
                # Step 6
                {'joint_gripper_finger_left':open},
                # Step 7
                {'joint_lift': 0.9,'wrist_extension': 0.0},
            ]
        # self.move_to_pose({'joint_lift': 0.9})
        # time.sleep(6)
        # self.move_to_pose({'joint_lift': [0.6,15.0]},custom_contact_thresholds=True)

        for i in range(len(poses)):
            # if i==3:
            #     print("detecting contact")
            #     self.lower_tool_until_contact()
            #     print("done")
            self.move_to_pose(poses[i])
            print(poses[i])
            rospy.loginfo("Completed calling : "+str(i))
            #sl = np.abs((-(self.goal_y))-0.26) * 10
            #rospy.loginfo("Sleeping for : "+str(time_sleep))
            time.sleep(3)
            rospy.loginfo("Completed performing step : "+str(i))

        return TriggerResponse(
            success=True,
            message='Completed object placement!'
            )
    # def lift_contact_func(effort, av_effort):
    #     single_effort_threshold = 20.0
    #     av_effort_threshold = 20.0
        
    #     if (effort <= single_effort_threshold):
    #         rospy.loginfo('Lift single effort less than single_effort_threshold: {0} <= {1}'.format(effort, single_effort_threshold))
    #     if (av_effort <= av_effort_threshold):
    #         rospy.loginfo('Lift average effort less than av_effort_threshold: {0} <= {1}'.format(av_effort, av_effort_threshold))
            
    #     return ((effort <= single_effort_threshold) or
    #             (av_effort < av_effort_threshold))

    # self.lift_down_contact_detector = GraspObjectNode(hm.get_lift_state, self.lift_contact_func)

    # def trigger_lower_until_contact_service_callback(self, request):
    #     direction_sign = -1
    #     lowest_allowed_m = 0.3
    #     success, message = self.lift_down_contact_detector.move_until_contact('joint_lift', lowest_allowed_m, direction_sign, self.move_to_pose)
    #     return TriggerResponse(
    #         success=success,
    #         message=message
    #         )

    def callback_goal(self,data):
        if not self.defined:
            print("############################################")
            print("defining goal locations")
            print("############################################")
            self.goal_x= data.point.x
            self.goal_y= data.point.y
            self.goal_z= data.point.z
            self.defined = True
        
    def main(self):
        
        hm.HelloNode.main(self, 'grasp_object', 'grasp_object', wait_for_first_pointcloud=False)
        self.joint_states_subscriber = rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)
        self.goal_subscriber = rospy.Subscriber("/goal_to_place",PointStamped, self.callback_goal)
        # self.flag_subscriber = rospy.Subscriber("/flag",Bool, self.callback_flag)
        # self.goal_subscriber_fingertip = rospy.Subscriber("/goal_loc_fingertip",PointStamped, self.callback_goal_fingertip)
        print("Subscribers called")
        self.trigger_place_object_service = rospy.Service('/grasp_object/trigger_grasp_object',
                                                           Trigger,
                                                           self.trigger_place_object_callback)

        # rospy.wait_for_service('/funmap/trigger_lower_until_contact')
        # rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_lower_until_contact.')
        # self.trigger_lower_until_contact_service = rospy.ServiceProxy('/funmap/trigger_lower_until_contact', 
        #                                                 Trigger, 
        #                                                 self.trigger_lower_until_contact_service_callback)
        # rospy.wait_for_service('/funmap/trigger_lower_until_contact')
        # rospy.loginfo('Node ' + self.node_name + ' /trigger_lower_until_contact.')
        # self.trigger_lower_until_contact_service = rospy.Service('/funmap/trigger_lower_until_contact',
        #                                                          Trigger,
        #                                                          self.trigger_lower_until_contact_service_callback)
        # self.trigger_lower_until_contact_service = rospy.Service('/funmap/trigger_lower_tool_until_contact',
                                                                #  Trigger)

                                                                
        print("After service")
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            rate.sleep()

# -----------------------------------------------------------------------------
    
if __name__ == '__main__':
    try:
        parser = ap.ArgumentParser(description='Grasp Object behavior for stretch.')
        args, unknown = parser.parse_known_args()
        node = GraspObjectNode()
        node.main()
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')   