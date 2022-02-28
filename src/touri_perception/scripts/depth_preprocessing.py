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
from std_msgs.msg import String
from sensor_msgs.msg import Image
import open3d
from ctypes import * # convert float to uint32
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, \
    (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# -----------------------------------------------------------------------------

def callback(ros_cloud):
    rospy.loginfo("-------Received ROS PointCloud2 message-------")
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, \
                                      field_names = field_names))

    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    rospy.loginfo("Converting ROS PointCloud2 ->  open3d format")
    
    # Set open3d_cloud
    if "rgb" in field_names:
        rospy.loginfo("Found an rgb point cloud")
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        
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
    
    rospy.loginfo("Finding a 2d plane")
    plane_model, inliers = open3d_cloud.segment_plane(distance_threshold=0.005,
                                         ransac_n=3,
                                         num_iterations=200)
    
    [a, b, c, d] = plane_model
    rospy.loginfo(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    rospy.loginfo("Getting the inliers")

    inlier_cloud = open3d_cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = open3d_cloud.select_by_index(inliers, invert=True)
    
    obb = inlier_cloud.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, obb])

    rospy.loginfo("Clustering objects")

    dists = open3d_cloud.compute_point_cloud_distance(inlier_cloud)
    dists = np.asarray(dists)
    ind = np.where((dists >= 0.005) & (dists <= 0.05))[0]
    cropped_pcd = open3d_cloud.select_by_index(ind)
    open3d.visualization.draw_geometries([cropped_pcd])
    
    with open3d.utility.VerbosityContextManager(
            open3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            cropped_pcd.cluster_dbscan(eps=0.02, min_points=100, print_progress=True))

    max_label = labels.max()
    rospy.loginfo(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cropped_pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
    open3d.visualization.draw_geometries([cropped_pcd])

    # return cropped_pcd

# -----------------------------------------------------------------------------

def main(args):
    # ROS node initialization
    rospy.init_node('test_pc_conversion_between_Open3D_and_ROS', anonymous=True)
    topic_name="/camera/depth/color/points"
    rospy.Subscriber(topic_name, PointCloud2, callback)      
    rospy.spin()
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main(sys.argv)