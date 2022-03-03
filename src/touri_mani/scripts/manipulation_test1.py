#!/usr/bin/env python2.7

# from cgitb import reset
# import rospy
# from control_msgs.msg import FollowJointTrajectoryGoal
# from trajectory_msgs.msg import JointTrajectoryPoint
# from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
# import hello_helpers.hello_misc as hm
# import hello_helpers.gripper_conversion as gripper
# import time
# import numpy as np
# import stretch_funmap.manipulation_planning as mp
# import cv2
from tkinter import N
import firebase_admin
from firebase_admin import credentials, db
# from __future__ import print_function

import rospy
import actionlib

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform, TransformStamped, PoseWithCovarianceStamped, PoseStamped, Pose, PointStamped
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseResult, MoveBaseFeedback
from nav_msgs.srv import GetPlan
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from tf.transformations import euler_from_quaternion
from tf2_geometry_msgs import do_transform_pose

import numpy as np
import scipy.ndimage as nd
import cv2
import math
import time
import threading
import sys
import os
import copy

import tf_conversions
import ros_numpy
import tf2_ros

import argparse as ap

import hello_helpers.hello_misc as hm
import hello_helpers.hello_ros_viz as hr

import stretch_funmap.merge_maps as mm
import stretch_funmap.navigate as nv
import stretch_funmap.mapping as ma
import stretch_funmap.segment_max_height_image as sm
import stretch_funmap.navigation_planning as na
import stretch_funmap.manipulation_planning as mp

import touri_planner

def create_map_to_odom_transform(t_mat):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = 'map'
    t.child_frame_id = 'odom'
    t.transform = ros_numpy.msgify(Transform, t_mat)
    return t

class ActuationNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.debug_directory = None


    def init_database():
        '''
        Init firebase database
        Uses a json key (not in the repo) to form a webSocket connection with the database
        '''
        cred = credentials.Certificate("/home/hello-robot/Desktop/TouRI/touri_ros/src/touri_core/keys/touri-65f07-firebase-adminsdk-wuv71-3751c21aa8.json")
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/'})


    def skill_req_listener(self, event):
        '''
        Event listener that reacts to changes in event db events
        Everytime an event is called, it sends a 
        # self.map_n_localize()
        # self.align_with_cliff()
        # self.dance()
        '''
        if(event.data["id"] != "none"):
            if(event.data["id" == 'localize']):
                self.map_n_localize()
            elif (event.data["id" == 'align_cliff']):
                self.align_with_cliff()
            elif (event.data["id" == 'dance']):
                self.dance()
            
            db.reference("/skill_req").update({"id" : "none", "inProgress" : False})




    def actuate(self, lift, extend, yaw = 0, grip = 0.05, base_linear = 0, base_rotation = 0):
        assert lift >= 0.2
        pose = {
                'joint_lift': lift,
                'wrist_extension': extend,
                # 'joint_wrist_yaw': yaw,
                # 'translate_mobile_base': base_linear,
                'rotate_mobile_base': base_rotation,
                # 'joint_gripper_finger_left' : grip,
                }

        self.move_to_pose(pose)
        

########################################################################################################################
    def map_n_localize(self):
        rospy.loginfo('Node ' + self.node_name + ' waiting to connect to /funmap/trigger_head_scan.')
        rospy.wait_for_service('/funmap/trigger_head_scan')
        rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_head_scan.')
        self.trigger_head_scan_service = rospy.ServiceProxy('/funmap/trigger_head_scan', Trigger)
        head_scan = TriggerRequest()
        result = self.trigger_head_scan_service(head_scan)


        rospy.wait_for_service('/funmap/trigger_global_localization')
        rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_global_localization')
        self.trigger_head_scan_service = rospy.ServiceProxy('/funmap/trigger_global_localization', Trigger)
        local_srv = TriggerRequest()
        result = self.trigger_head_scan_service(local_srv)

########################################################################################################################

    def align_with_cliff(self):
        rospy.loginfo('Node ' + self.node_name + ' waiting to connect to /funmap/trigger_align_with_nearest_cliff.')
        rospy.wait_for_service('/funmap/trigger_align_with_nearest_cliff')
        rospy.loginfo('Node ' + self.node_name + ' connected to /funmap/trigger_align_with_nearest_cliff')
        self.trigger_head_scan_service = rospy.ServiceProxy('/funmap/trigger_align_with_nearest_cliff', Trigger)
        cliff_srv = TriggerRequest()
        result = self.trigger_head_scan_service(cliff_srv)


########################################################################################################################

    def dance(self):
        self.actuate(lift=0.3, extend=0, yaw=0.5, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.4, extend=0.3, yaw=0, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.45, extend=0, yaw=0.5, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.5, extend=0.3, yaw=0, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.55, extend=0, yaw=0.5, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.6, extend=0.2, yaw=0, grip=0.05, base_linear=5, base_rotation=3)

    

    def correct_robot_pose(self, original_robot_map_pose_xya, corrected_robot_map_pose_xya):
        # Compute and broadcast the corrected transformation from
        # the map frame to the odom frame.
        print('original_robot_map_pose_xya =', original_robot_map_pose_xya)
        print('corrected_robot_map_pose_xya =', corrected_robot_map_pose_xya)
        x_delta = corrected_robot_map_pose_xya[0] - original_robot_map_pose_xya[0]
        y_delta = corrected_robot_map_pose_xya[1] - original_robot_map_pose_xya[1]
        ang_rad_correction = hm.angle_diff_rad(corrected_robot_map_pose_xya[2], original_robot_map_pose_xya[2])
        c = np.cos(ang_rad_correction)
        s = np.sin(ang_rad_correction)
        rot_mat = np.array([[c, -s], [s, c]])
        x_old, y_old, a_old = original_robot_map_pose_xya
        xy_old = np.array([x_old, y_old])
        tx, ty = np.matmul(rot_mat, -xy_old) + np.array([x_delta, y_delta]) + xy_old
        t = np.identity(4)
        t[0,3] = tx
        t[1,3] = ty
        t[:2,:2] = rot_mat
        self.map_to_odom_transform_mat = np.matmul(t, self.map_to_odom_transform_mat)
        self.tf2_broadcaster.sendTransform(create_map_to_odom_transform(self.map_to_odom_transform_mat))
        
    def publish_corrected_robot_pose_markers(self, original_robot_map_pose_xya, corrected_robot_map_pose_xya):
        # Publish markers to visualize the corrected and
        # uncorrected robot poses on the map.
        timestamp = rospy.Time.now()
        markers = MarkerArray()
        ang_rad = corrected_robot_map_pose_xya[2]
        x_axis = [np.cos(ang_rad), np.sin(ang_rad), 0.0]
        x, y, a = corrected_robot_map_pose_xya
        point = [x, y, 0.1]
        rgba = [0.0, 1.0, 0.0, 0.5]
        m_id = 0
        m = hr.create_sphere_marker(point, m_id, 'map', timestamp, rgba=rgba, diameter_m=0.1, duration_s=0.0)
        markers.markers.append(m)
        m_id += 1
        m = hr.create_axis_marker(point, x_axis, m_id, 'map', timestamp, rgba, length=0.2, arrow_scale=3.0)
        markers.markers.append(m)
        m_id += 1
        x, y, a = original_robot_map_pose_xya
        point = [x, y, 0.1]
        rgba = [1.0, 0.0, 0.0, 0.5]
        m = hr.create_sphere_marker(point, m_id, 'map', timestamp, rgba=rgba, diameter_m=0.1, duration_s=0.0)
        markers.markers.append(m)
        m_id += 1
        m = hr.create_axis_marker(point, x_axis, m_id, 'map', timestamp, rgba, length=0.2, arrow_scale=3.0)
        markers.markers.append(m)
        m_id += 1
        self.marker_array_pub.publish(markers)
        
    def perform_head_scan(self, fill_in_blindspot_with_second_scan=True, localize_only=False, global_localization=False, fast_scan=False):
        node = self
        
        # trigger_request = TriggerRequest() 
        # trigger_result = self.trigger_d435i_high_accuracy_mode_service(trigger_request)
        # rospy.loginfo('trigger_result = {0}'.format(trigger_result))
            
        # Reduce the occlusion due to the arm and grabber. This is
        # intended to be run when the standard grabber is not holding
        # an object.
        ma.stow_and_lower_arm(node)

        # Create and perform a new full scan of the environment using
        # the head.
        head_scan = ma.HeadScan(voi_side_m=16.0)
        head_scan.execute_full(node, fast_scan=fast_scan)

        scaled_scan = None
        scaled_merged_map = None

        # Save the new head scan to disk.
        if self.debug_directory is not None:
            dirname = self.debug_directory + 'head_scans/'
            # If the directory does not already exist, create it.
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = 'head_scan_' + hm.create_time_string()
            head_scan.save(dirname + filename)
        else:
            rospy.loginfo('FunmapNode perform_head_scan: No debug directory provided, so debugging data will not be saved.')

        head_scan.make_robot_footprint_unobserved()
        save_merged_map = False
        
        if self.merged_map is None:
            # The robot does not currently have a map, so initialize
            # the map with the new head scan.
            rospy.loginfo('perform_head_scan: No map available, so setting the map to be the scan that was just taken.')
            self.merged_map = head_scan
            robot_pose = [head_scan.robot_xy_pix[0], head_scan.robot_xy_pix[1], head_scan.robot_ang_rad]
            self.robot_poses.append(robot_pose)
            self.localized = True
            save_merged_map = True
        else:
            if localize_only and (not global_localization):
                # The scan was performed to localize the robot locally.
                rospy.loginfo('perform_head_scan: Performing local localization.')
                use_full_size_scans = False
                if use_full_size_scans: 
                    affine_matrix, original_robot_map_pose, corrected_robot_map_pose = mm.estimate_scan_1_to_scan_2_transform(head_scan,
                                                                                                                              self.merged_map, 
                                                                                                                              display_on=False,
                                                                                                                              show_unaligned=False,
                                                                                                                              full_localization=False,
                                                                                                                              init_target=None,
                                                                                                                              grid_search=False,
                                                                                                                              small_search=False)
                else: 
                    original_robot_map_frame_pose, corrected_robot_map_frame_pose, original_robot_map_image_pose, corrected_robot_map_image_pose, scaled_scan, scaled_merged_map = ma.localize_with_reduced_images(head_scan, self.merged_map, global_localization=False, divisor=2, small_search=True)

                    corrected_robot_map_pose = corrected_robot_map_frame_pose
                    original_robot_map_pose = original_robot_map_frame_pose
                    # Save the scaled scans to disk for debugging.
                    if self.debug_directory is not None:
                        dirname = self.debug_directory + 'scaled_localization_scans/'
                        # If the directory does not already exist, create it.
                        if not os.path.exists(dirname):
                            os.makedirs(dirname)
                        time_string = hm.create_time_string()
                        filename = 'localization_scaled_head_scan_' + time_string
                        scaled_scan.save(dirname + filename)
                        filename = 'localization_scaled_merged_map_' + time_string
                        scaled_merged_map.save(dirname + filename)
                    else:
                        rospy.loginfo('FunmapNode perform_head_scan: No debug directory provided, so debugging data will not be saved.')
                self.localized = True
            elif (not self.localized) or (localize_only and global_localization):
                # The robot has not been localized with respect to the
                # current map or the scan was performed solely to
                # globally localize the robot. This attempts to
                # localize the robot on the map by reducing the sizes
                # of the scan and the map in order to more efficiently
                # search for a match globally.

                # This does not merge the new scan into the current map.
                rospy.loginfo('perform_head_scan: Performing global localization.')
                save_merged_map = False
                
                original_robot_map_frame_pose, corrected_robot_map_frame_pose, original_robot_map_image_pose, corrected_robot_map_image_pose, scaled_scan, scaled_merged_map = ma.localize_with_reduced_images(head_scan, self.merged_map, global_localization=True, divisor=6) #4)
                corrected_robot_map_pose = corrected_robot_map_frame_pose
                original_robot_map_pose = original_robot_map_frame_pose
                self.localized = True

                # Save the scaled scans to disk for debugging.
                if self.debug_directory is not None:
                    dirname = self.debug_directory + 'scaled_localization_scans/'
                    # If the directory does not already exist, create it.
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    time_string = hm.create_time_string()
                    filename = 'localization_scaled_head_scan_' + time_string
                    scaled_scan.save(dirname + filename)
                    filename = 'localization_scaled_merged_map_' + time_string
                    scaled_merged_map.save(dirname + filename)
                else:
                    rospy.loginfo('FunmapNode perform_head_scan: No debug directory provided, so debugging data will not be saved.')
            else: 
                # The robot has been localized with respect to the
                # current map, so proceed to merge the new head scan
                # into the map. This assumes that the robot's
                # estimated pose is close to its actual pose in the
                # map. It constrains the matching optimization to a
                # limited range of positions and orientations.
                rospy.loginfo('perform_head_scan: Performing local map merge.')
                original_robot_map_pose, corrected_robot_map_pose = mm.merge_scan_1_into_scan_2(head_scan, self.merged_map)
                save_merged_map = True
                
            # Store the corrected robot pose relative to the map frame.
            self.robot_poses.append(corrected_robot_map_pose)

            self.correct_robot_pose(original_robot_map_pose, corrected_robot_map_pose)
            pub_robot_markers = True
            if pub_robot_markers: 
                self.publish_corrected_robot_pose_markers(original_robot_map_pose, corrected_robot_map_pose)
                
        if save_merged_map: 
            # If the merged map has been updated, save it to disk.
            if self.debug_directory is not None:
                head_scans_dirname = self.debug_directory + 'head_scans/'
                # If the directory does not already exist, create it.
                if not os.path.exists(head_scans_dirname):
                    os.makedirs(head_scans_dirname)
                merged_maps_dirname = self.debug_directory + 'merged_maps/'
                # If the directory does not already exist, create it.
                if not os.path.exists(merged_maps_dirname):
                    os.makedirs(merged_maps_dirname)
                time_string = hm.create_time_string()
                if scaled_scan is not None: 
                    filename = 'localization_scaled_head_scan_' + time_string
                    scaled_scan.save(head_scans_dirname + filename)
                if scaled_merged_map is not None: 
                    filename = 'localization_scaled_merged_map_' + time_string
                    scaled_merged_map.save(merged_maps_dirname + filename)
                filename = 'merged_map_' + hm.create_time_string()
                self.merged_map.save(merged_maps_dirname + filename)
            else:
                rospy.loginfo('FunmapNode perform_head_scan: No debug directory provided, so debugging data will not be saved.')


        if fill_in_blindspot_with_second_scan and (not localize_only):
            # Turn the robot to the left in attempt to fill in its
            # blindspot due to its mast.
            turn_ang = (70.0/180.0) * np.pi
            
            # Command the robot to turn to point to the next
            # waypoint.
            rospy.loginfo('robot turn angle in degrees =' + str(turn_ang * (180.0/np.pi)))
            at_goal = self.move_base.turn(turn_ang, publish_visualizations=True)
            if not at_goal:
                message_text = 'Failed to reach turn goal.'
                rospy.loginfo(message_text)
            self.perform_head_scan(fill_in_blindspot_with_second_scan=False)



########################################################################################################################
    def main(self):
        hm.HelloNode.main(self, 'stow_command', 'stow_command', wait_for_first_pointcloud=False)
        # #DB
        # self.init_database()
        # db.reference("/skill_req").listen(self.skill_req_listener)
        #self.debug_directory = rospy.get_param('/funmap/debug_directory')
        # self.merged_map = None
        # self.robot_poses = []
        # self.move_base = nv.MoveBase(self, self.debug_directory)
        # self.map_to_odom_transform_mat = np.identity(4)
        # self.tf2_broadcaster = tf2_ros.TransformBroadcaster()

        # self.marker_array_pub = rospy.Publisher('/funmap/marker_array', MarkerArray, queue_size=1)

        #self.map_n_localize()
        # self.align_with_cliff()
        # rospy.loginfo("performing head scan")
        # self.perform_head_scan()
        # rospy.loginfo("done head scan")
        self.dance()
        rospy.logdebug("#################################################################################")
        rospy.logdebug("#################################################################################")
        rospy.logdebug("#################################################################################")
        rospy.logdebug("#################################################################################")
        rospy.logdebug("#################################################################################")
        rospy.logdebug("#################################################################################")
        rospy.logdebug("#################################################################################")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()



if __name__ == '__main__':
    try:
        node = ActuationNode()
        node.main()
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')