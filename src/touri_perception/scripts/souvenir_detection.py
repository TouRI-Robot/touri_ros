#!/usr/bin/env python3
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel, Shivani Sivakumar"
__mail__      = "jkpatel@andrew.cmu.edu, ssivaku3@andrew.cmu.edu"
__copyright__ = "NONE"

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Twist, Vector3
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from ctypes import * # convert float to uint32
from std_srvs.srv import Trigger, TriggerRequest

# Touri Imports
from touri_perception.srv import transform_service, transform_serviceResponse
from touri_perception.srv import perception_picking, perception_pickingResponse
from touri_perception.srv import picking_centroid_calc

# Socket Imports
import open3d as o3d
import random
import os
import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
from std_msgs.msg import ColorRGBA

# PDB Imports 
from pdb import set_trace as bp

# -----------------------------------------------------------------------------

bridge = CvBridge()
# ip_address = "192.168.0.11"
ip_address = "172.26.10.200"
# ip_address = "172.26.246.68"
# '172.26.246.68' - Jigar when connected to CMU Secure

# -----------------------------------------------------------------------------

class final:
    def __init__(self):
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        ts = message_filters.TimeSynchronizer([depth_sub, image_sub], 10)
        ts.registerCallback(self.image_update_callback)

        # Publish a marker
        self.marker_pub = rospy.Publisher('souvenir_test', Marker)

        self.image_content = None
        self.depth_content = None
        self.x_3d = 0
        self.y_3d = 0
        self.z_3d = 0
        # Perception picking service
        self.image_server = rospy.Service('perception_picking_service', perception_picking, self.service_callback)
        rospy.wait_for_service('picking_centroid_calc')
        self.centroid_client = rospy.ServiceProxy('picking_centroid_calc', picking_centroid_calc)

    def make_marker(self, marker_type, scale):
        # make a visualization marker array for the occupancy grid
        m = Marker()
        m.action = Marker.ADD
        m.header.frame_id = 'base_link'
        m.header.stamp = rospy.Time.now()
        m.ns = 'marker_test_%d' % marker_type
        m.id = 0
        m.type = marker_type
        m.pose.orientation.y = 0
        m.pose.orientation.w = 1
        m.pose.position.x = self.x_3d;
        m.pose.position.y = self.y_3d;
        m.pose.position.z = self.z_3d;
        m.scale = scale
        m.color.r = 1.0;
        m.color.g = 0.1;
        m.color.b = 0.1;
        m.color.a = 0.7;
        return m

    def image_update_callback(self, depth_image, image):
        self.image_content = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.depth_content = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(depth_image.height, depth_image.width, -1)
        scale = Vector3(0.025, 0.025, 0.025)
        self.marker_pub.publish(self.make_marker(Marker.SPHERE,   scale))
    
    def service_callback(self, req):
        rospy.loginfo("Perception picking service callback")
        image = self.image_content
        
        # ============ Server to send the image for inference ============
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_socket.connect((ip_address, 8485))
        connection = client_socket.makefile('wb')
        img_counter = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        frame = image
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame, 0)
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)
        
        # ============ Client to receive the detections ============ 
        HOST=''
        PORT=8485
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST,PORT))
        s.listen(10)
        conn,addr=s.accept()

        data = b""
        payload_size = struct.calcsize(">L")
        
        while len(data) < payload_size:
            # print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        # print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

        # ============ Output extract prediction data ============
        outputs = frame
        preds = outputs[0]
        scores = outputs[1]
        classes = outputs[2]
        result = np.where(classes == req.object_id)
        obj_id = result[0][0]
        
        try:
            x_start = int(preds[obj_id][0])
            y_start = int(preds[obj_id][1])
            x_end = int(preds[obj_id][2])
            y_end = int(preds[obj_id][3])
            centroid_x = int((x_start + x_end)/2)
            centroid_y = int((y_start + y_end)/2)
            
            image = cv2.circle(image, (centroid_x, centroid_y), 5, (255,0,255), -1)
            cv2.imwrite("/home/hello-robot/trial/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/scripts/centroid_img.jpg",image)
            
            detected = True
             
            # ============ Client for calculating centroid from C++ pipeline ============
            resp_centroid = self.centroid_client(x_start, y_start, x_end, y_end)
            print("resp_centroid : ",resp_centroid)
            x_3d = resp_centroid.x_reply
            y_3d = resp_centroid.y_reply
            z_3d = resp_centroid.z_reply
            width = resp_centroid.width 

            rospy.wait_for_service('transform_to_base')
            try:
                transform_to_base_client = rospy.ServiceProxy('transform_to_base', transform_service)
            except rospy.ServiceException as e:
                    print("Service call failed: %s"%e)
                    return
            resp1 = transform_to_base_client(x_3d, y_3d, z_3d)

            x_base = resp1.x_reply
            y_base = resp1.y_reply
            z_base = resp1.z_reply

            self.x_3d  = x_base
            self.y_3d  = y_base
            self.z_3d  = z_base
            # self.image_pub.publish(self.br.cv2_to_imgmsg(image))
            # self.depth_pub.publish(self.br.cv2_to_imgmsg(depth_data))

            detected = True
            print("CENTROID IN BASE", x_base, y_base, z_base)
            
            return perception_pickingResponse(detected,x_base,y_base,z_base)                
        except Exception as e:
            print("not detected",e)
            detected = False
            return perception_pickingResponse(detected,-1,-1,-1)
        

def main(args):
    rospy.init_node('souvenir_pose_estimator', anonymous=True)
    f = final()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
        

