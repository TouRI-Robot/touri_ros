#!/usr/bin/env python3
# ROS Imports 
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
import cv2
import open3d as o3d
import random
import os
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32

import sys
from std_msgs.msg import String, Bool
from touri_perception.srv import transform_service,transform_serviceResponse
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Twist, Vector3
from std_srvs.srv import Trigger, TriggerRequest
from touri_perception.srv import perception_shipping,perception_shippingResponse, centroid_calc

# Socket Imports
import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib

from pdb import set_trace as bp

bridge = CvBridge()


convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, \
    (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
class final:
    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)
        # self.depth_pub = rospy.Publisher("depth_topic_2",Image)
        # self.pointcloud_pub = rospy.Publisher("pointcloud_topic_2",PointCloud2)
        # self.flag_pub= rospy.Publisher("/flag",Bool, queue_size=10)
        # self.pose_pub = rospy.Publisher('/detected_point', PointStamped, queue_size=10)

        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        cloud_sub = message_filters.Subscriber("/camera/depth/color/points", PointCloud2)
        ts = message_filters.TimeSynchronizer([depth_sub, image_sub, cloud_sub], 10)

        ts.registerCallback(self.callback)

        self.image_content = None
        self.depth_content = None
        self.pointcloud_content = None

        # self.pose_pub = rospy.Publisher('/detected_point', PointStamped, queue_size=10)
        # self.flag = False
        self.image_server = rospy.Service('perception_shipping_service', perception_shipping, self.service_callback)
        
        rospy.wait_for_service('centroid_calc')
        self.centroid_client = rospy.ServiceProxy('centroid_calc', centroid_calc)
            
    def callback(self, depth_image, image, pointcloud):
        # print(" callabck")

        self.image_content = image
        self.depth_content = depth_image
        self.pointcloud_content = pointcloud


    def service_callback(self, req):
        print("service callback")
        image = self.image_content
        depth_image = self.depth_content
        pointcloud = self.pointcloud_content

        print("Image : ",type(image))
        # Detectron2
        image = bridge.imgmsg_to_cv2(image, "bgr8")
        
        # Send image to PC
        # outputs = predictor(image)
        # Recieve outputs
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_socket.connect(('172.26.246.68', 8486))
        connection = client_socket.makefile('wb')
        img_counter = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        frame = image
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        #  data = zlib.compress(pickle.dumps(frame, 0))
        data = pickle.dumps(frame, 0)
        size = len(data)
        print("{}: {}".format(img_counter, size))
        client_socket.sendall(struct.pack(">L", size) + data)
            # img_counter += 1

        # cam.release()
        # connection.close()

        HOST=''
        PORT=8486
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        print('Socket created')
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST,PORT))
        print('Socket bind complete')
        s.listen(10)
        print('Socket now listening')

        conn,addr=s.accept()

        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))

        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

        print("received frame",frame)
        outputs = frame
        print("received frame",outputs)
        preds = outputs[0]
        scores = outputs[1]
        classes = outputs[2]

        # v = Visualizer(image[:, :, ::-1],
        #                 metadata=test_metadata, 
        #                 scale=0.8
        #                 )
        # print("Outputs : ",outputs)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # im = out.get_image()[:, :, ::-1]
        
        # Output from PC

        # detected = True
        # bp()
        # cv2.imshow('RealSense', im)
        # cv2.waitKey(1)

        # print("depth_image : ",type(depth_image))
        # bp()
        try:
            x_start = int(preds[0][0])
            y_start = int(preds[0][1])
            x_end = int(preds[0][2])
            y_end = int(preds[0][3])
            detected = True
            resp_centroid = self.centroid_client(x_start, y_start,x_end,y_end)
            x_3d = resp_centroid.x_reply
            y_3d = resp_centroid.y_reply
            z_3d = resp_centroid.z_reply
            width = resp_centroid.width
            print("printing values")

            # # 3D
            # print("starting 3D pipeline")
            # # start_point1 = (619,186)
            # # end_point1 = (960,630)
            # start_point1 = (x_start,y_start)
            # end_point1 = (x_end,y_end)
            
            # w = end_point1[0] - start_point1 [0]
            # h = end_point1[1] - start_point1 [1]
            # centroidx = int(w/2 + start_point1[0])
            # centroidy = int(h/2 + start_point1[1])

            # print("scaling")

            # # scale bb
            # scaling_factor = 1.2
            # scaled_w = scaling_factor * w
            # scaled_h = scaling_factor * h
            # start_point2 = (int(centroidx-scaled_w/2),int(centroidy-scaled_h/2))
            # end_point2 = (int(centroidx+scaled_w/2),int(centroidy+scaled_h/2))
            # color1 = (255, 0, 0)
            # color2 = (0, 255, 0)
            # thickness = 2
            # print("changing image")
            # image = cv2.rectangle(image, start_point1, end_point1, color1, thickness)
            # image = cv2.rectangle(image, start_point2, end_point2, color2, thickness)

            # # intrinsic matrix
            # camera_matrix = np.array([[910.7079467773438, 0.0, 634.5316772460938], [0.0, 910.6213989257812, 355.40097045898], [0.0, 0.0, 1.0]])

            # f_x = camera_matrix[0,0]
            # c_x = camera_matrix[0,2]
            # f_y = camera_matrix[1,1]
            # c_y = camera_matrix[1,2]

            # # 3D point from 2D point
            # start_homo = np.array([[start_point2[0]],[start_point2[1]],[1.0]])
            # start_3d = np.linalg.inv(camera_matrix) @ start_homo
            # end_homo = np.array([[end_point2[0]],[end_point2[1]],[1.0]])
            # end_3d = np.linalg.inv(camera_matrix) @ end_homo

            # z = 0.75
            # start_x_3d = ((start_point2[0] - c_x) / f_x) * z
            # start_y_3d = ((start_point2[1] - c_y) / f_y) * z
            # end_x_3d = ((end_point2[0] - c_x) / f_x) * z
            # end_y_3d = ((end_point2[1] - c_y) / f_y) * z

            # # visualize 3D pcl

            # rospy.loginfo("-------Received ROS PointCloud2 message-------")
            
            # # Get cloud data from pointcloud
            # field_names=[field.name for field in pointcloud.fields]
            # cloud_data = list(pc2.read_points(pointcloud, skip_nans=True, \
            #                                 field_names = field_names))

            # # Check if pointcloud is empty
            # open3d_cloud = o3d.geometry.PointCloud()
            # if len(cloud_data)==0:
            #     rospy.loginfo("Converting an empty cloud")
            #     return None

            # rospy.loginfo("Converting ROS PointCloud2 ->  open3d format")
            
            # # Set open3d_cloud
            # if "rgb" in field_names:
            #     rospy.loginfo("Found an rgb point cloud")
            #     IDX_RGB_IN_FIELD=3 # x, y, z, rgb        
            #     # Get xyzopen3d_cloudopen3d_cloudopen3d_cloud
            #     xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)
            #     # Get rgb Check whether int or float
            #     if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            #         rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
            #     else:
            #         rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
            #     # combine
            #     open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
            #     open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
            # else:
            #     xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
            #     open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
            #     # points = np.asarray(open3d_cloud.points)
            #     # colors = np.asarray(open3d_cloud.colors)
            

            # # open3d_cloud = o3d.io.read_point_cloud("/home/shivani0812/Desktop/scripts/1664987632821457.pcd", format='pcd')
            # # o3d.visualization.draw_geometries([open3d_cloud])

            # # remove points from pointcloud
            # points = np.asarray(open3d_cloud.points)
            # colors = np.asarray(open3d_cloud.colors)
            # open3d_cloud = open3d_cloud.select_by_index(np.where((start_x_3d<= points[:,0]) & (points[:,0] <= end_x_3d) & (start_y_3d <= points[:,1]) & (points[:,1] <= end_y_3d))[0])

            # print("Poincloud  : ",type(open3d_cloud))
            # # centroid = np.mean(points,axis=0).reshape(1,-1)
            # # print(centroid)
            # # plane_center = open3d_cloud.get_center()
            # # plane_center = np.reshape(plane_center.T, (1,3))
            # # points = np.append(points, plane_center, axis=0)
            # # colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)
            # # o3d.visualization.draw_geometries([open3d_cloud])

            # # plane segmentation
            # print("Downsampling point cloud")
            # open3d_cloud = open3d_cloud.voxel_down_sample(voxel_size = 0.003)
            # new_cloud = open3d_cloud
            # outlier_cloud = open3d_cloud
            # centroids = []
            # # normals = []

            # # for i in range(5):
            # #     plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02,
            # #                                         ransac_n=3,
            # #                                         num_iterations=500)
                
            # #     [a, b, c, d] = plane_model
            # #     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            # #     print("Getting the inliers")

            # #     inlier_cloud = outlier_cloud.select_by_index(inliers)
            # #     outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

            # #     inlier_points = np.asarray(inlier_cloud.points)
            # #     centroid = np.mean(inlier_points,axis=0).reshape(1,-1)
            # #     centroids.append(centroid)
            # #     print(f"Centroid: {centroid}")
            # #     plane_center = inlier_cloud.get_center()    
            # #     plane_center = np.reshape(plane_center.T,(1,3))
            # #     print(f"Plane center: {plane_center}")

            # #     points = np.append(points, plane_center, axis=0)
            # #     colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)

            # #     pcl = o3d.geometry.PointCloud()
            # #     pcl.points = o3d.utility.Vector3dVector(points)
            # #     pcl.colors = o3d.utility.Vector3dVector(colors)
            # #     # print(outlier_cloud.has_normals())
                
            # #     # bb on segmented plane
            # #     # aabb = plane_cloud.get_axis_aligned_bounding_box()
            # #     # aabb.color = (1,0,0)
            # #     obb = inlier_cloud.get_oriented_bounding_box()
            # #     obb.color = (0,1,0)
            # #     box_coords = np.asarray(obb.get_box_points())
            # #     obb_center = obb.get_center()
            # #     print(f"OBB center: {obb_center}")
            # #     #find normal to plane
            # #     # normal = np.array([a,b,c])
            # #     # normal = normal / np.linalg.norm(normal)
            # #     # print(f"Normal: {normal}")
            # #     # normals.append(normal)
            # #     # normal = np.reshape(normal.T,(1,3))
            # #     # points = np.append(points, normal, axis=0)
            # #     colors = np.append(colors, np.array([[0,1,0]]), axis=0)
                



            # #     # print(box_coords)
            # #     # print(obb.has_normals())

            # #     x,y,z = centroid.flatten()
            # #     for i in range(box_coords.shape[0]):
            # #         x,y,z = box_coords[i].flatten()

            # #     x,y,z = centroid.flatten()



            # #     # o3d.visualization.draw_geometries([open3d_cloud, pcl, obb])
            # #     # n = pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # #     # print("n", n)


            # #     # o3d.geometry.estimate_normals(
            # #     # inlier_cloud,
            # #     # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
            # #     #                                                   max_nn=30))
            # #     # o3d.visualization.draw_geometries([inlier_cloud])
        
            # #     o3d.visualization.draw_geometries([outlier_cloud, pcl, obb])
            # normals = []
            # for i in range(4):
            #     plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.02,
            #                                         ransac_n=3,
            #                                         num_iterations=500)
                
            #     [a, b, c, d] = plane_model

            #     plane_cloud = outlier_cloud.select_by_index(inliers)
            #     outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)

            #     points = np.asarray(plane_cloud.points)
            #     colors = np.asarray(plane_cloud.colors)
            #     plane_center = plane_cloud.get_center()
            #     plane_center = np.reshape(plane_center.T,(1,3))
            #     centroids.append(plane_center)
            #     points = np.append(points, plane_center, axis=0)
            #     colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)

            #     pcl = o3d.geometry.PointCloud()
            #     pcl.points = o3d.utility.Vector3dVector(points)
            #     pcl.colors = o3d.utility.Vector3dVector(colors)

            #     #find normal to plane
            #     normal = np.array([a,b,c])
            #     normal = normal / np.linalg.norm(normal)
            #     print(f"Normal: {normal}")
            #     # normal = np.reshape(normal.T,(1,3))
            #     normals.append(normal)
            #     normals_arr = np.array(normals)
            #     # np.append(normals,normal, axis = 0)
            #     # print("normal:", normal)

            #     # bb on segmented plane
            #     # aabb = plane_cloud.get_axis_aligned_bounding_box()
            #     # aabb.color = (1,0,0)
            #     obb = plane_cloud.get_oriented_bounding_box()
            #     obb.color = (0,1,0)
            #     o3d.visualization.draw_geometries([open3d_cloud, pcl, obb])
            
            # # for n in normals:
            # #     print("normals are:", n)
            # print(normals_arr)
            # a = 0
            # b = 0
            # print("calculating dot product")
            # for i in range(normals_arr.shape[0]):
            #     print("i:", i)
            #     for j in range(normals_arr.shape[0]):
            #         print("j:", i)
            #         if(i!=j):
            #             dot_product = np.dot(normals_arr[i],normals_arr[j])
            #             if abs(dot_product)<=1 and abs(dot_product)>=0.96:
            #                 a = i
            #                 b = j
            #                 break
            # print("a", a)
            # print("b", b)

            # # for c in centroids:
            # #     print(c)
            # print(centroids[a])
            # print(centroids[b])

            # diff = abs(centroids[a] - centroids[b])
            # print("difference:", diff)
            # max = 0
            # index = 0
            # # for i in range(diff.shape[1]):
            # #     if diff[0][i] >= max:
            # #         max = diff[0][i]
            # #         index = i
            # # centroid3d = diff[index] / 2

            # # centroid3d = [0,0,0]
            # # centroid3d[index] = diff[0][index] / 2
            # cent1 = np.array(centroids[a])
            # cent2 = np.array(centroids[b])
            # print(cent1.shape)
            # print(cent2.shape)
            # # for i in range(3):
            # #     if(i!=index):
            # centroid3d= (cent1+ cent2)/2     
                    
            # print("final centroid:", centroid3d)
            # final_centroid = np.array(centroid3d).reshape((1,3))
            # print("reshaped final centroid")



            # points = np.asarray(new_cloud.points)
            # colors = np.asarray(new_cloud.colors)
            # points = np.append(points, np.array(centroid3d).reshape((1,3)), axis=0)
            # colors = np.append(colors, np.array([[1.0,0,0]]), axis=0)
            # new_cloud.points = o3d.utility.Vector3dVector(points)
            # new_cloud.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([new_cloud])

            # x_3d = final_centroid[0][0]
            # y_3d = final_centroid[0][1]
            # z_3d = final_centroid[0][2]

            print(x_3d, y_3d, z_3d)

            # point  = PointStamped()
            # point.header.stamp = rospy.Time.now()
            # point.header.frame_id = '/map'
            # point.point.x = x_3d
            # point.point.y = y_3d
            # point.point.z = z_3d
            print("waiting for service")
            rospy.wait_for_service('transform_to_base')
            print("service found")

            try:
                transform_to_base_client = rospy.ServiceProxy('transform_to_base', transform_service)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
                return
            resp1 = transform_to_base_client(x_3d, y_3d, z_3d)
            x_base = resp1.x_reply
            y_base = resp1.y_reply
            z_base = resp1.z_reply
            print(x_base, y_base, z_base)

            # self.image_pub.publish(self.br.cv2_to_imgmsg(image))
            # self.depth_pub.publish(self.br.cv2_to_imgmsg(depth_image))
            # self.pointcloud_pub.publish(pointcloud)
            detected = True


            # b = Bool()
            # b.data = self.flag
            # self.flag_pub.publish(b)
            # detected = true
            print("base:")
            print(x_base, y_base, z_base)
            point  = PointStamped()
            point.header.stamp = rospy.Time.now()
            point.header.frame_id = '/map'
            point.point.x = resp1.x_reply
            point.point.y = resp1.y_reply
            point.point.z = resp1.z_reply
            # place_pub.publish(point)


            # import os
            # time.sleep(1)
            # cmd = "rosservice call /grasp_object/trigger_grasp_object"
            # os.system("gnome-terminal -- {}".format(cmd))

            # print(detected,x_base,y_base,z_base)
            print("called the service")

            return perception_shippingResponse(detected,x_base,y_base,z_base)
            # return resp1
            # return perceptionResponse(detected,x_base,y_base,z_base)
        except IndexError:
            detected = False
            return perception_shippingResponse(detected,-1,-1,-1)


def main(args):
    rospy.init_node('final_centroid_calc', anonymous=True)
    f = final()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)