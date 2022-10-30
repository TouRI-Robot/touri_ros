#!/usr/bin/env python3
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel"
__mail__      = "jkpatel@andrew.cmu.edu"
__copyright__ = "NONE"

# -----------------------------------------------------------------------------

from csv import excel_tab
import cv2
import numpy as np
import rospy
import time
import sys
import open3d
from matplotlib import pyplot as plt
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Twist, Vector3
from touri_perception.srv import transform_service,transform_serviceResponse
from std_srvs.srv import Trigger, TriggerRequest

# -----------------------------------------------------------------------------

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, \
    (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# -----------------------------------------------------------------------------
displayed_pointcloud = False

def callback(ros_cloud):
    global displayed_pointcloud
    rospy.loginfo("-------Received ROS PointCloud2 message-------")
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, \
                                      field_names = field_names))

    # Check if pointcloud is empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data)==0:
        rospy.loginfo("Converting an empty cloud")
        return None

    rospy.loginfo("Converting ROS PointCloud2 ->  open3d format")
    
    # Set open3d_cloud
    if "rgb" in field_names:
        rospy.loginfo("Found an rgb point cloud")
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)
        # Get rgb Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        # combine
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
    
    rospy.loginfo("Downsampling point cloud")
    open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size = 0.003)
    outlier_cloud = open3d_cloud
    
    # Finding a horizontal plane to place the object
    centroids = []
    for i in range(5):
        rospy.loginfo("Finding a 2d plane")
        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02,
                                            ransac_n=3,
                                            num_iterations=200)
        
        [a, b, c, d] = plane_model
        rospy.loginfo(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        
        rospy.loginfo("Getting the inliers")

        inlier_cloud = outlier_cloud.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
        inlier_points = np.asarray(inlier_cloud.points)
        centroid = np.mean(inlier_points,axis=0).reshape(1,-1)
        obb = inlier_cloud.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        box_coords = np.asarray(obb.get_box_points())
        centroids.append(centroid)
        rospy.wait_for_service('transform_to_base')

        try:
            transform_to_base_client = rospy.ServiceProxy('transform_to_base', transform_service)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            return
        x,y,z = centroid.flatten()
        resp1 = transform_to_base_client(x, y, z)
        z_plane = resp1.z_reply
        # print("z plane  :",z_plane)
        # open3d.visualization.draw_geometries([inlier_cloud,open3d_cloud, obb])
        is_place_plane = True
        for i in range(box_coords.shape[0]):
            x,y,z = box_coords[i].flatten()
            resp1 = transform_to_base_client(x, y, z)
            # z_val = resp1.z_reply
            # # print("z_val : ",z_val)
            # if not (z_val>0.4 and z_val<0.85):
            #     is_place_plane = False
            #     break
                
        if not is_place_plane:
            continue

        x,y,z = centroid.flatten()
        resp1 = transform_to_base_client(x, y, z)
        print("resp1 : ",resp1)
        z_plane = resp1.z_reply
        if (z_plane>0.4 and z_plane<0.8):
            rospy.loginfo(f"Z value of the plane : {z_val}.")
            if not displayed_pointcloud:
                displayed_pointcloud = True
                open3d.visualization.draw_geometries([inlier_cloud,open3d_cloud, obb])
                point  = PointStamped()
                point.header.stamp = rospy.Time.now()
                point.header.frame_id = '/map'
                point.point.x = resp1.x_reply
                point.point.y = resp1.y_reply
                point.point.z = resp1.z_reply
                place_pub.publish(point)
                
                import os
                time.sleep(1)
                cmd = "rosservice call /grasp_object/trigger_grasp_object"
                os.system("gnome-terminal -- {}".format(cmd))
            # print("wartint gorr service")
            # rospy.wait_for_service('/grasp_object/trigger_grasp_object')
            # sos_service = rospy.ServiceProxy('/grasp_object/trigger_grasp_object', Trigger)
            # # Create an object of the type TriggerRequest. We nned a TriggerRequest for a Trigger service
            # sos = TriggerRequest()
            # # Now send the request through the connection
            # result = sos_service(sos)
                print("called")
            return resp1

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # ROS node initialization
    rospy.loginfo("Starting plane detection Node")
    rospy.init_node('detect_plane_node', anonymous=True)
    topic_name ="/camera/depth/color/points"
    place_pub  = rospy.Publisher("/goal_to_place",PointStamped, queue_size=1)
    rospy.Subscriber(topic_name, PointCloud2, callback)
    rospy.spin()
    
# -----------------------------------------------------------------------------
