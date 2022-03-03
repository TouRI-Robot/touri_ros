import numpy as np
import scipy.ndimage as nd
import scipy.signal as si
import cv2
import skimage as sk
import math
import stretch_funmap.max_height_image as mh
import stretch_funmap.segment_max_height_image as sm
import stretch_funmap.ros_max_height_image as rm
import hello_helpers.hello_misc as hm
import ros_numpy as rn
import rospy
import os

import stretch_funmap.merge_maps as mm
import stretch_funmap.navigate as nv
import stretch_funmap.mapping as ma
import stretch_funmap.segment_max_height_image as sm
import stretch_funmap.navigation_planning as na
import stretch_funmap.manipulation_planning as mp


from stretch_funmap.numba_manipulation_planning import numba_find_base_poses_that_reach_target, numba_check_that_tool_can_deploy
from stretch_funmap.numba_check_line_path import numba_find_contact_along_line_path, numba_find_line_path_on_surface



class TouriPlanner():

    def __init__(self, tf2_buffer):
        self.tf2_buffer = tf2_buffer
        # How far to look ahead.
        look_ahead_distance_m = 2.0
        # Robot's width plus a safety margin.
        look_to_side_distance_m = 1.3
        m_per_pix = 0.006
        pixel_dtype = np.uint8 
        robot_head_above_ground = 1.13
        lowest_distance_below_ground = 0.03
        voi_height_m = robot_head_above_ground + lowest_distance_below_ground

        robot_right_edge_m = 0.2
        voi_side_x_m = 2.0 * look_to_side_distance_m
        voi_side_y_m = look_ahead_distance_m

        voi_axes = np.identity(3)
        voi_origin = np.array([-(voi_side_x_m/2.0), -(voi_side_y_m + robot_right_edge_m), -lowest_distance_below_ground])

        # Define the VOI using the base_link frame
        old_frame_id = 'base_link'
        voi = rm.ROSVolumeOfInterest(old_frame_id, voi_origin, voi_axes, voi_side_x_m, voi_side_y_m, voi_height_m)
        # Convert the VOI to the map frame to handle mobile base changes
        new_frame_id = 'map'
        lookup_time = rospy.Time(0) # return most recent transform
        timeout_ros = rospy.Duration(0.1)
        stamped_transform =  tf2_buffer.lookup_transform(new_frame_id, old_frame_id, lookup_time, timeout_ros)
        points_in_old_frame_to_new_frame_mat = rn.numpify(stamped_transform.transform)
        voi.change_frame(points_in_old_frame_to_new_frame_mat, new_frame_id)

        self.voi = voi
        self.max_height_im = rm.ROSMaxHeightImage(self.voi, m_per_pix, pixel_dtype)
        self.max_height_im.print_info()
        self.updated = False


    def plan_to_reach(self, reach_xyz_pix, robot_xya_pix=None, floor_mask=None):
        # This is intended to perform coarse positioning of the
        # gripper near a target 3D point.
        robot_reach_xya_pix = None
        wrist_extension_m = None

        i_x, i_y, i_z = reach_xyz_pix

        max_height_im = self.max_height_im
        # Check if a map exists
        # if self.merged_map is None:
        #     message = 'No map exists yet, so unable to plan a reach.'
        #     rospy.logerr(message)
        #     return None, None

        if robot_xya_pix is None: 
            robot_xy_pix, robot_ang_rad, timestamp = max_height_im.get_robot_pose_in_image(self.tf2_buffer)
            robot_xya_pix = [robot_xy_pix[0], robot_xy_pix[1], robot_ang_rad]


        rospy.loginfo('###############################################################################################')
        rospy.loginfo('###############################################################################################')
        rospy.loginfo('###############################################################################################')
        rospy.loginfo(robot_xya_pix)

        rospy.loginfo('###############################################################################################')
        rospy.loginfo('###############################################################################################')
        rospy.loginfo('###############################################################################################')


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

        return robot_reach_xya_pix, simple_reach_plan




