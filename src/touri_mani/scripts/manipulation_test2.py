#!/usr/bin/env python2.7
import sys
from os.path import dirname
sys.path.append(dirname("/home/hello-robot/stretch_ros/stretch_funmap"))

from tkinter import N
import firebase_admin
from firebase_admin import credentials, db
from math import cos, sin
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

def print_divider():
    print("\n#######################################################\n")


def divided_print(input_val):
    print("\n#######################################################")
    print(input_val)
    print("#######################################################\n")


class ManipulationNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.debug_directory = None


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

    def dance(self):
        self.actuate(lift=0.3, extend=0, yaw=0.5, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.4, extend=0.3, yaw=0, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.45, extend=0, yaw=0.5, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.5, extend=0.3, yaw=0, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.55, extend=0, yaw=0.5, grip=0.05, base_linear=5, base_rotation=3)
        self.actuate(lift=0.6, extend=0.2, yaw=0, grip=0.05, base_linear=5, base_rotation=3)
########################################################################################################################
    

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

########################################################################################################################

    #TODO: Remove
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
    
########################################################################################################################
    #TODO: Remove
    def publish_nav_plan_markers(self, line_segment_path, image_to_points_mat, clicked_frame_id):
        path_height_m = 0.2
        points = [np.matmul(image_to_points_mat, np.array([p[0], p[1], path_height_m, 1.0]))[:3] for p in line_segment_path]
        points = [[p[0], p[1], path_height_m] for p in points]
        self.publish_path_markers(points, clicked_frame_id)
        
########################################################################################################################

    def plan_to_reach(self, reach_xyz_pix, robot_xya_pix=None, floor_mask=None):
        # This is intended to perform coarse positioning of the
        # gripper near a target 3D point.
        robot_reach_xya_pix = None
        wrist_extension_m = None

        i_x, i_y, i_z = reach_xyz_pix
        
        max_height_im = self.merged_map.max_height_im
        # Check if a map exists
        if self.merged_map is None:
            message = 'No map exists yet, so unable to plan a reach.'
            rospy.logerr(message)
            return None, None
        
        if robot_xya_pix is None: 
            robot_xy_pix, robot_ang_rad, timestamp = max_height_im.get_robot_pose_in_image(self.tf2_buffer)
            robot_xya_pix = [robot_xy_pix[0], robot_xy_pix[1], robot_ang_rad]
        
        end_xy_pix = np.int64(np.round(np.array([i_x, i_y])))
        m_per_height_unit = max_height_im.m_per_height_unit
        # move the gripper to be above the target point
        extra_target_height_m = 0.01
        target_z = i_z + (extra_target_height_m / m_per_height_unit)
        target_z_m = target_z * m_per_height_unit
        target_xyz_pix = (end_xy_pix[0], end_xy_pix[1], target_z)

        image_display_on = False

        manipulation_planner = mp.ManipulationPlanner()
        base_x_pix, base_y_pix, base_ang_rad, wrist_extension_m = manipulation_planner.base_pose(max_height_im,
                                                                                                 target_xyz_pix,
                                                                                                 robot_xya_pix,
                                                                                                 image_display_on=image_display_on)
        if image_display_on: 
            c = cv2.waitKey(0)
            
        if base_x_pix is None:
            rospy.logerr('No valid base pose found for reaching the target.')
            return None, None

        robot_reach_xya_pix = [base_x_pix, base_y_pix, base_ang_rad]

        base_link_point = max_height_im.get_pix_in_frame(np.array(reach_xyz_pix), 'base_link', self.tf2_buffer)
        
        simple_reach_plan = []
        
        # close the gripper
        simple_reach_plan.append({'joint_gripper_finger_left': 0.0})

        # move the lift to be at the height of the target
        # The fingers of the gripper touch the floor at a joint_lift
        # height of 0.0 m, so moving the lift link to the height of
        # the target will result in the fingers being at the height of
        # the target.
        height_m = base_link_point[2]
        safety_z_m = 0.0
        simple_reach_plan.append({'joint_lift': height_m + safety_z_m})
        
        # rotate the gripper to be in the center
        # of the swept volume of the wrist (a
        # little right of center when looking out
        # from the robot to the gripper)
        #simple_reach_plan.append({'joint_gripper': -0.25})
        simple_reach_plan.append({'joint_wrist_yaw': -0.25})
        
        # reach out to the target
        # Reach to a point that is not fully at the target.
        safety_reach_m = 0.1 # 10cm away from the target
        simple_reach_plan.append({'wrist_extension': wrist_extension_m - safety_reach_m})
        print("simple_reach_plan" , simple_reach_plan)
        return robot_reach_xya_pix, simple_reach_plan

########################################################################################################################

    def plan_a_path(self, end_xy_pix, robot_xya_pix=None, floor_mask=None):
        # Transform the robot's current estimated pose as represented
        # by TF2 to the map image. Currently, the estimated pose is
        # based on the transformation from the map frame to the
        # base_link frame, which is updated by odometry and
        # corrections based on matching head scans to the map.
        path = None
        
        # Check if a map exists
        if self.merged_map is None:
            message = 'No map exists yet, so unable to drive to a good scan spot.'
            rospy.loginfo('Plan a path: No map exists')
            return path, message

        max_height_im = self.merged_map.max_height_im
        rospy.loginfo('Plan a path: Getting bots xya')
        if robot_xya_pix is None: 
            robot_xy_pix, robot_ang_rad, timestamp = max_height_im.get_robot_pose_in_image(self.tf2_buffer)
            robot_xya_pix = [robot_xy_pix[0], robot_xy_pix[1], robot_ang_rad]

        rospy.loginfo('Plan a path: Current XY -> {}'.format(robot_xy_pix))
        max_height_im = self.merged_map.max_height_im
        rospy.loginfo('Plan a path: Planning a path')
        line_segment_path, message = na.plan_a_path(max_height_im, robot_xya_pix,
                                                    end_xy_pix, floor_mask=floor_mask)
        rospy.loginfo('Plan a path: Done planning')
        return line_segment_path, message
    
########################################################################################################################
   
    def navigate_to_map_pixel(self, end_xy, end_angle=None, robot_xya_pix=None, floor_mask=None):
        # Set the D435i to Default mode for obstacle detection
        trigger_request = TriggerRequest() 
        trigger_result = self.trigger_d435i_default_mode_service(trigger_request)
        rospy.loginfo('trigger_result = {0}'.format(trigger_result))

        # Move the head to a pose from which the D435i can detect
        # obstacles near the front of the mobile base while moving
        # forward.
        rospy.loginfo('navigate_to_map_pixel: Move base')
        self.move_base.head_to_forward_motion_pose()

        rospy.loginfo('navigate_to_map_pixel: Plan path')
        line_segment_path, message = self.plan_a_path(end_xy, robot_xya_pix=robot_xya_pix, floor_mask=floor_mask)
        if line_segment_path is None:
            success = False
            rospy.loginfo('navigate_to_map_pixel: {}'.format(message))
            return success, message
        rospy.loginfo('navigate_to_map_pixel: Done planning')
        # Existence of the merged map is checked by plan_a_path, but
        # to avoid future issues I'm introducing this redundancy.
        if self.merged_map is None:
            success = False
            rospy.loginfo('navigate_to_map_pixel: Done map available!')
            return success, 'No map available for planning and navigation.'
        max_height_im = self.merged_map.max_height_im
        map_frame_id = self.merged_map.max_height_im.voi.frame_id
                
        # Query TF2 to obtain the current estimated transformation
        # from the map image to the map frame.
        rospy.loginfo('navigate_to_map_pixel: Query TF2')
        image_to_points_mat, ip_timestamp = max_height_im.get_image_to_points_mat(map_frame_id, self.tf2_buffer)
        
        if image_to_points_mat is not None:                

            # Publish a marker array to visualize the line segment path.
            # self.publish_nav_plan_markers(line_segment_path, image_to_points_mat, map_frame_id)

            # Iterate through the vertices of the line segment path,
            # commanding the robot to drive to them in sequence using
            # in place rotations and forward motions.
            successful = True
            for p0, p1 in zip(line_segment_path, line_segment_path[1:]):
                # Query TF2 to obtain the current estimated transformation
                # from the image to the odometry frame.
                image_to_odom_mat, io_timestamp = max_height_im.get_image_to_points_mat('odom', self.tf2_buffer)

                # Query TF2 to obtain the current estimated transformation
                # from the robot's base_link frame to the odometry frame.
                robot_to_odom_mat, ro_timestamp = hm.get_p1_to_p2_matrix('base_link', 'odom', self.tf2_buffer)

                # Navigation planning is performed with respect to a
                # odom frame height of 0.0, so the heights of
                # transformed points are 0.0. The simple method of
                # handling the heights below assumes that the odom
                # frame is aligned with the floor, so that ignoring
                # the z coordinate is approximately equivalent to
                # projecting a point onto the floor.
                
                # Convert the current and next waypoints from map
                # image pixel coordinates to the odom
                # frame. 
                p0 = np.array([p0[0], p0[1], 0.0, 1.0])
                p0 = np.matmul(image_to_odom_mat, p0)[:2]
                p1 = np.array([p1[0], p1[1], 0.0, 1.0])
                next_point_xyz = np.matmul(image_to_odom_mat, p1)
                p1 = next_point_xyz[:2]

                # Find the robot's current pose in the odom frame.
                xya, timestamp = self.get_robot_floor_pose_xya()
                r0 = xya[:2]
                r_ang = xya[2]
                
                # Check how far the robot's current location is from
                # its current waypoint. The current waypoint is where
                # the robot would ideally be located.
                waypoint_tolerance_m = 0.25
                waypoint_error = np.linalg.norm(p0 - r0)
                rospy.loginfo('waypoint_error =' + str(waypoint_error))
                if waypoint_error > waypoint_tolerance_m:
                    message_text = 'Failed due to waypoint_error being above the maximum allowed error.'
                    rospy.loginfo(message_text)
                    success=False
                    message=message_text
                    return success, message

                # Find the angle in the odometry frame that would
                # result in the robot pointing at the next waypoint.
                travel_vector = p1 - r0
                travel_dist = np.linalg.norm(travel_vector)
                travel_ang = np.arctan2(travel_vector[1], travel_vector[0])
                rospy.loginfo('travel_dist =' + str(travel_dist))
                rospy.loginfo('travel_ang =' + str(travel_ang * (180.0/np.pi)))

                # Find the angle that the robot should turn in order
                # to point toward the next waypoint.
                turn_ang = hm.angle_diff_rad(travel_ang, r_ang)

                # Command the robot to turn to point to the next
                # waypoint.
                rospy.loginfo('robot turn angle in degrees =' + str(turn_ang * (180.0/np.pi)))
                at_goal = self.move_base.turn(turn_ang, publish_visualizations=True)
                if not at_goal:
                    message_text = 'Failed to reach turn goal.'
                    rospy.loginfo(message_text)
                    success=False
                    message=message_text
                    return success, message
                    
                # The head seems to drift sometimes over time, such
                # that the obstacle detection region is no longer
                # observed resulting in false positives. Hopefully,
                # this will correct the situation.
                self.move_base.head_to_forward_motion_pose()

                # FOR FUTURE DEVELOPMENT OF LOCAL NAVIGATION
                testing_future_code = False
                if testing_future_code: 
                    check_result = self.move_base.check_line_path(next_point_xyz, 'odom')
                    rospy.loginfo('Result of check line path = {0}'.format(check_result))
                    local_path, local_path_frame_id = self.move_base.local_plan(next_point_xyz, 'odom')
                    if local_path is not None:
                        rospy.loginfo('Found local path! Publishing markers for it!')
                        self.publish_path_markers(local_path, local_path_frame_id)
                    else:
                        rospy.loginfo('Did not find a local path...')
                
                # Command the robot to move forward to the next waypoing. 
                at_goal = self.move_base.forward(travel_dist, publish_visualizations=False)
                if not at_goal:
                    message_text = 'Failed to reach forward motion goal.'
                    rospy.loginfo(message_text)
                    success=False
                    message=message_text
                    return success, message
                
                rospy.loginfo('Turn and forward motion succeeded.')

            if end_angle is not None:
                # If a final target angle has been provided, rotate
                # the robot to match the target angle.
                rospy.loginfo('Attempting to achieve the final target orientation.')
                
                # Find the robot's current pose in the map frame. This
                # assumes that the target angle has been specified
                # with respect to the map frame.
                xya, timestamp = self.get_robot_floor_pose_xya(floor_frame='map')
                r_ang = xya[2]
            
                # Find the angle that the robot should turn in order
                # to point toward the next waypoint.
                turn_ang = hm.angle_diff_rad(end_angle, r_ang)

                # Command the robot to turn to point to the next
                # waypoint.
                rospy.loginfo('robot turn angle in degrees =' + str(turn_ang * (180.0/np.pi)))
                at_goal = self.move_base.turn(turn_ang, publish_visualizations=True)
                if not at_goal:
                    message_text = 'Failed to reach turn goal.'
                    rospy.loginfo(message_text)
                    success=False
                    message=message_text
                    return success, message
            
        success=True
        message='Completed drive to new scan location.'
        return success, message
    
########################################################################################################################
    # TODO: Refector
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
        #? Change based on use case
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
    def print_current_XY(self):
        max_height_im = self.merged_map.max_height_im
        robot_xy_pix, robot_ang_rad, timestamp = max_height_im.get_robot_pose_in_image(self.tf2_buffer)

        wrist_frame = 'link_aruco_top_wrist'
        wrist_points_to_image_mat, ip_timestamp = max_height_im.get_points_to_image_mat(wrist_frame, self.tf2_buffer)
        wrist_x, wrist_y, wrist_z = wrist_points_to_image_mat[:, 3][:3]

        #! REMOVE START
        print("X: {}".format(robot_xy_pix[0]))
        print("Y: {}".format(robot_xy_pix[1]))
        print("angle: {}".format(robot_ang_rad))
        print("Wrist X: {}".format(wrist_x))
        print("Wrist Y: {}".format(wrist_y))
        #! REMOVE END

        # print_divider()
        # print("X: {}".format(robot_xy_pix[0]))
        # print("Y: {}".format(robot_xy_pix[1]))
        # print("Z: {}".format(wrist_z))
        # print("Wrist X: {}".format(wrist_x))
        # print("Wrist Y: {}".format(wrist_y))
        # print_divider()


    def transform_xyz_to_xyz_pix(self, x_m, y_m, z_m):
        max_height_im = self.merged_map.max_height_im

        robot_xy_pix, robot_ang_rad, timestamp = max_height_im.get_robot_pose_in_image(self.tf2_buffer)
        wrist_frame = 'link_aruco_top_wrist'
        wrist_points_to_image_mat, ip_timestamp = max_height_im.get_points_to_image_mat(wrist_frame, self.tf2_buffer)
        wrist_x, wrist_y, wrist_z = wrist_points_to_image_mat[:, 3][:3]


        current_x_pix = robot_xy_pix[0]
        current_y_pix = robot_xy_pix[1]
        current_z_pix = wrist_z

        m_per_pix = max_height_im.m_per_pix
        m_per_height_unit = max_height_im.m_per_height_unit

        dest_x_pix = (x_m / m_per_pix) + current_x_pix
        dest_y_pix = (y_m / m_per_pix) + current_y_pix
        dest_z_pix = (z_m / m_per_height_unit) + current_z_pix

        final_dest_man = np.array([dest_x_pix, dest_y_pix, dest_z_pix ])
        return final_dest_man
        


########################################################################################################################    
    
    def main(self):
        hm.HelloNode.main(self, 'stow_command', 'stow_command', wait_for_first_pointcloud=False)
 
        self.debug_directory = rospy.get_param('/funmap/debug_directory')
        self.merged_map = None
        self.robot_poses = []
        self.move_base = nv.MoveBase(self, self.debug_directory)
        self.map_to_odom_transform_mat = np.identity(4)
        self.tf2_broadcaster = tf2_ros.TransformBroadcaster()

        self.marker_array_pub = rospy.Publisher('/funmap/marker_array', MarkerArray, queue_size=1)
        default_service = '/camera/switch_to_default_mode'
        self.trigger_d435i_default_mode_service = rospy.ServiceProxy(default_service, Trigger)

        # self.perform_head_scan(fill_in_blindspot_with_second_scan=True)  
        
        while True:
            start = input('Start localizing: ')  
            # self.perform_head_scan(fill_in_blindspot_with_second_scan=False, fast_scan=True, localize_only=True, global_localization=True)    
            self.perform_head_scan(fill_in_blindspot_with_second_scan=True, fast_scan=False, global_localization=False)    
            self.print_current_XY()

            # x_dest = input('Enter X(m): ')  
            # y_dest = input('Enter Y(m): ') 
            # z_dest = input('Enter Z(m): ')
            #! REMOVE START
            x_delta = input('Enter X(m): ')  
            y_delta = input('Enter Y(m): ') 
            angle = input('Enter angle(rad): ')

            divided_print("STARTING NAVIGATING")

            delta_xy = np.array([x_delta, y_delta])

            rot_m = np.array(([cos(angle), -sin(angle)], [sin(angle), cos(angle)]))

            delta_xy = np.matmul(rot_m, delta_xy)

            max_height_im = self.merged_map.max_height_im
            robot_xy_pix, robot_ang_rad, timestamp = max_height_im.get_robot_pose_in_image(self.tf2_buffer)

            robot_xy_pix = robot_xy_pix + delta_xy

            divided_print(robot_xy_pix)

            success, message = self.navigate_to_map_pixel(end_xy= robot_xy_pix, end_angle=robot_ang_rad)
            divided_print("DONE NAVIGATING")
            #! REMOVE END
            # final_dest_man = self.transform_xyz_to_xyz_pix(x_dest, y_dest, z_dest)
            # divided_print(final_dest_man)

            # divided_print("PLANNING")
            # dest_xya, mani_plan = self.plan_to_reach(final_dest_man)
            # divided_print(dest_xya)
            # divided_print("DONE PLANNING")

            # divided_print("STARTING NAVIGATING")
            # success, message = self.navigate_to_map_pixel(end_xy=dest_xya[:2], end_angle =dest_xya[2])
            # divided_print("DONE NAVIGATING")

            # if success:
            #     for pose in mani_plan:
            #         self.move_to_pose(pose)
            #         divided_print("DONE MANIPULATING")
                
            # else:
            #     rospy.loginfo(" Error, cannot reach")
            #     divided_print("FAILED NAVIGATING")
        


if __name__ == '__main__':
    try:
        node = ManipulationNode()
        node.main()
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')
        

