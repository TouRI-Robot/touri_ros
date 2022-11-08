#!/usr/bin/env python3
# ROS Imports 
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from ctypes import * # convert float to uint32
import sys
from std_msgs.msg import String, Bool
from touri_perception.srv import transform_service,transform_serviceResponse
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Twist, Vector3
from std_srvs.srv import Trigger, TriggerRequest
from touri_perception.srv import perception_shipping,perception_shippingResponse, centroid_calc

# System Imports
import numpy as np
import cv2
import open3d as o3d
import random
import os

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
        # Defining subcribers 
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_update_callback)
        self.image_content = None

        # Perception dropping service
        self.image_server = rospy.Service('perception_shipping_service', perception_shipping, self.service_callback)
        
        # TODO : What is service failure
        rospy.wait_for_service('centroid_calc')
        self.centroid_client = rospy.ServiceProxy('centroid_calc', centroid_calc)

    def image_update_callback(self, data):
        self.image_content = bridge.imgmsg_to_cv2(data)

    def service_callback(self, req):
        rospy.loginfo("Perception dropping service callback")
        image = self.image_content
        print("Image : ", type(image), image.shape)
        # image = bridge.imgmsg_to_cv2(image, "bgr8")
        
        # ============ Server to send the image for inference ============
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_socket.connect(('172.26.246.68', 8486))
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
        PORT=8486
        s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST,PORT))
        s.listen(10)
        conn,addr=s.accept()
        data = b""
        payload_size = struct.calcsize(">L")
        # print("payload_size: {}".format(payload_size))

        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        # print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        # print("msg_size: {}".format(msg_size))
        
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        
        # ============ Output extract bounding box dims ============
        outputs = frame
        preds = outputs[0]
        scores = outputs[1]
        classes = outputs[2]
        try:
            detected = True
            x_start = int(preds[0][0])
            y_start = int(preds[0][1])
            x_end = int(preds[0][2])
            y_end = int(preds[0][3])

            # ============ Client for calculating centroid from C++ pipeline ============
            resp_centroid = self.centroid_client(x_start, y_start,x_end,y_end)
            x_3d = resp_centroid.x_reply
            y_3d = resp_centroid.y_reply
            z_3d = resp_centroid.z_reply
            width = resp_centroid.width
            
            # ============ Convert centroid from 3D to base frame ============
            print("XYZ in camera frame",x_3d, y_3d, z_3d)
            
            rospy.wait_for_service('transform_to_base')
            try:
                transform_to_base_client = rospy.ServiceProxy('transform_to_base', transform_service)
            except rospy.ServiceException as e:
                detected = False
                return perception_shippingResponse(detected,-1,-1,-1)

            resp1 = transform_to_base_client(x_3d, y_3d, z_3d)
            x_base = resp1.x_reply
            y_base = resp1.y_reply
            z_base = resp1.z_reply
            
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
            return perception_shippingResponse(detected,x_base,y_base,z_base)
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