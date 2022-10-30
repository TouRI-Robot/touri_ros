#!/usr/bin/env python3

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
        
    def trigger_drop_object_callback(self, request):
        rospy.loginfo("drop service called")
        pose = {'joint_head_tilt':(np.deg2rad(-90))/3}
        self.move_to_pose(pose)
        # rospy.wait_for_service('perception_service')
        # try:
        #     perception_client = rospy.ServiceProxy('perception_service', perception)
        # except rospy.ServiceException as e:
        #     print("Service call failed: %s"%e)
        #     return
        
        # Step 0 : Moving the camera to detect object 
        i = 1
        time.sleep(5)
        # flag = False
        # while flag==False:
        #     resp1 = perception_client(True)
        #     flag = resp1.detected
        #     if flag:
        #         self.goal_x = resp1.x_reply
        #         self.goal_y = resp1.y_reply
        #         self.goal_z = resp1.z_reply
        #         break
            
        #     rospy.loginfo("Value:"+str(flag))
        rospy.loginfo("Rotating clockwise : "+str(i*20)+"by 20 degrees")
        # rot_angle = np.deg2rad(-20) * i
        pose = {'joint_head_pan': np.deg2rad(-20) * i}
        i=i+1
        if i>9:
            # print("i : ",i)
            rospy.loginfo("Rotating clockwise : "+str((i-9)*20)+"by 10 degrees")
            pose = {'joint_head_pan': np.deg2rad(20) * (i-9)}
        rospy.loginfo(pose)
        
        if(i>13):
            # print("i : ",i)
            rospy.loginfo("Couldn't detect object")
            pose = {'joint_head_pan': 0}
            self.move_to_pose(pose)
            return
        self.move_to_pose(pose)
        rospy.loginfo("Done moving")
        time.sleep(3)

        open = 0.15283721447
        close=-0.37
            
        # Steps
        if self.goal_x and self.goal_y and self.goal_z:
            poses = [
                # Step 0 - Home position
                {'joint_lift': 0.3,'wrist_extension': 0.0, 'joint_gripper_finger_left':close},
                # Step 1 - Rotate to bring the cup in center
                {'rotate_mobile_base': np.arctan(self.goal_y/np.abs(self.goal_x))},
                # Step 2 - Rotate Head
                {'joint_head_pan':0},
                # Step 3 - Translate
                {'translate_mobile_base': self.goal_x - 0.6},
                # Step 4
                {'joint_lift': self.goal_z},
                # {'joint_lift': 0.9},
                # Step 5
                {'rotate_mobile_base': 1.57},
                # Step 6
                {'joint_head_pan':-1.57},
                
                # Step 7
                {'translate_mobile_base': self.goal_x},
                # Step 8
                {'wrist_extension': (-(self.goal_y)) - 0.36},
                # Step 9
                {'joint_gripper_finger_left':open},
                # Step 10
                {'joint_lift':0.9},
                # Step 11
                {'wrist_extension':0},
                # Step 12
                {'joint_head_tilt':-1.57*2/3, 'joint_head_pan':0}
            ]
        # 3,7,8,10
        for i in range(len(poses)):
            rospy.loginfo("===================================================================")
            rospy.loginfo("Performing step : "+str(i)+" : "+str(poses[i]))
            if (i==0):
                time_sleep = 2
            if (i==1):
                time_sleep = 0.05 * (np.rad2deg(np.arctan(self.goal_y/self.goal_x)))
            if(i==2):
                time_sleep = 2
            if(i==3):
                error = 1000
                while(np.abs(error) > 0.01):
                    # resp1 = perception_client(True)
                    # if(resp1.detected==False):
                    #     for i in range(5):
                    #         resp1 = perception_client(True)
                    #         if(resp1.detected==True):
                    #             break
                    #     else:
                    #         print("Not detected")
                    #         return
                    # self.goal_x = resp1.x_reply
                    # self.goal_y = resp1.y_reply
                    # self.goal_z = resp1.z_reply
                    # error = self.goal_x - 0.6
                    # rospy.loginfo("Correcting error : "+str(error))
                    # poses[i] = {'translate_mobile_base':error}
                    # self.move_to_pose(poses[i])
                    # time.sleep(5*(np.abs(error)))

                    # resp1 = perception_client(True)
                    # if(resp1.detected==False):
                    #     for i in range(5):
                    #         resp1 = perception_client(True)
                    #         if(resp1.detected==True):
                    #             break
                    #     else:
                    #         print("Not detected")
                    #         return
                    self.goal_x = resp1.x_reply
                    self.goal_y = resp1.y_reply
                    self.goal_z = resp1.z_reply
                    error = self.goal_x - 0.6
                rospy.loginfo("Error after step 3 : "+str(error))
                continue
            if(i==4):
                # resp1 = perception_client(True)
                # if(resp1.detected==False):
                #     for i in range(5):
                #         resp1 = perception_client(True)
                #         if(resp1.detected==True):
                #             break
                #     else:
                #         print("Not detected")
                #         return
                # self.goal_x = resp1.x_reply
                # self.goal_y = resp1.y_reply
                # self.goal_z = resp1.z_reply
                poses[i] = {'joint_lift': self.goal_z}
                # poses[i] = {'joint_lift': 0.9}
                time_sleep = 4
            if (i==5 or i==6):
                time_sleep = 3
            if(i==7):
                error = 1000
                while(np.abs(error) > 0.006):
                    # resp1 = perception_client(True)
                    # if(resp1.detected==False):
                    #     for i in range(5):
                    #         resp1 = perception_client(True)
                    #         if(resp1.detected==True):
                    #             break
                    #     else:
                    #         print("Not detected")
                    #         return
                    # self.goal_x = resp1.x_reply
                    # self.goal_y = resp1.y_reply
                    # self.goal_z = resp1.z_reply
                    error = self.goal_x #+ 0.03
                    rospy.loginfo("Correcting error : "+str(error))
                    poses[i] = {'translate_mobile_base':error}
                    self.move_to_pose(poses[i])
                    time.sleep(5*(np.abs(error)))

                    # resp1 = perception_client(True)
                    # if(resp1.detected==False):
                    #     for i in range(5):
                    #         resp1 = perception_client(True)
                    #         if(resp1.detected==True):
                    #             break
                    #     else:
                    #         print("Not detected")
                    #         return
                    # self.goal_x = resp1.x_reply
                    # self.goal_y = resp1.y_reply
                    # self.goal_z = resp1.z_reply
                    error = self.goal_x #+ 0.03
                rospy.loginfo("Error after step 7 : "+str(error))
                # error = self.goal_x #+ 0.03
                # rospy.loginfo("error : "+str(error))
                # while(np.abs(error) > 0.006):
                #     # if error>0:
                    
                #     rospy.loginfo("Correcting error : "+str(error))
                #     poses[i] = {'translate_mobile_base': error}
                #     self.move_to_pose(poses[i])
                #     time.sleep(5*(np.abs(error)))
                #     error = self.goal_x #+ 0.03
                # rospy.loginfo("Error after step 7 : "+str(error))
                continue
            if(i==8):
                rospy.loginfo("self.goal_y"+str(self.goal_y))
                extension = (-(self.goal_y))-0.26 + 0.055
                rospy.loginfo("extension : "+str(extension))
                # while(self.goal_y > 0.01):
                    # print("extension",extension)
                poses[i] = {'wrist_extension':extension}
                if extension > 0.5:
                    extension = 0.5
                self.move_to_pose(poses[i])
                time.sleep(10*(extension))
                # self.move_to_pose(poses[i])
                    # extension = (-(self.goal_y))-0.36
                continue
            if(i==9 or i==10):
                time_sleep = 3
            self.move_to_pose(poses[i])
            # rospy.loginfo("Completed calling : "+str(i))
            sl = np.abs((-(self.goal_y))-0.26) * 10
            # rospy.loginfo("Sleeping for : "+str(time_sleep))
            time.sleep(np.abs(time_sleep))
            rospy.loginfo("Completed performing step : "+str(i))

            if(i==11 or i==12):
                time_sleep = 2

        return TriggerResponse(
            success=True,
            message='Completed object [placement]!'
            )

    # def callback_goal(self,data):
    #     self.defined = True
    #     self.goal_x= data.point.x
    #     self.goal_y= data.point.y
    #     self.goal_z= data.point.z

    # def callback_flag(self,data):
    #     self.detected = data
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
        # self.goal_subscriber = rospy.Subscriber("/goal_loc",PointStamped, self.callback_goal)
        # self.flag_subscriber = rospy.Subscriber("/flag",Bool, self.callback_flag)
        self.trigger_grasp_object_service = rospy.Service('/grasp_object/trigger_grasp_object',
                                                           Trigger,
                                                           self.trigger_grasp_object_callback)
        self.goal_subscriber = rospy.Subscriber("/goal_to_place",PointStamped, self.callback_goal)

        rospy.loginfo("After service - Service server started")
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