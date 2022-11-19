#!/usr/bin/env python3

from __future__ import print_function
import rospy
from std_srvs.srv import Trigger, TriggerResponse
import time
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
import os
import sys
import keyboard
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from touri_perception.srv import tap_service, tap_serviceResponse
from touri_mani.srv import grasp_service, grasp_serviceResponse
import hello_helpers.hello_misc as hm


# Socket Imports
import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
import time
from datetime import datetime

from pdb import set_trace as bp
imagepath = "temp_img"
imageIter = 0

bridge = CvBridge()

ip_address = "172.26.10.200" # MRSD CMPTR
# ip_address = "172.26.246.68" # Jigar
# '172.26.246.68' - Jigar when connected to CMU Secure

touri_keys = "/home/hello-robot/trial/touri_integration/keys/touri-65f07-firebase-adminsdk-wuv71-b245c875f8.json"

cred = credentials.Certificate(touri_keys)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/',
    'storageBucket' : 'touri-65f07.appspot.com' 
    })

def uploadImg(imgPath):
    bucket = storage.bucket()
    blob = bucket.blob(imgPath)
    blob.upload_from_filename(imgPath)
    blob.make_public()
    
    db.reference("autoSkills/pickPlace").update({
        "imgSrc" : blob.public_url
    })
    print("blob",blob.public_url)

def sendFinishMessage(msg: str):
    try : 
        cred = credentials.Certificate("keys/touri-65f07-firebase-adminsdk-wuv71-b245c875f8.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/',
            'storageBucket' : 'touri-65f07.appspot.com' 
        })
    except:
        pass  
    
    db.reference("state").update({
        "botMsg" : msg,
        "runningAutoCommand" : False,
    })

class final:
    def __init__(self):
        self.bridge = CvBridge()
        # Defining subcribers 
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_update_callback)
        self.image_content = None
        self.detections = None
        self.send_img_service = rospy.Service('/send_image_service', Trigger, self.trigger_response)
        self.tap_img_service = rospy.Service('/tap_service', tap_service, self.call_grasp_service)
        self.results_pub = rospy.Publisher('/results_image', Image, queue_size=10) 
        
    def call_grasp_service(self, request):
        try:
            x_tap = request.x_tap
            y_tap = request.y_tap
            print("x_tap : ",x_tap)
            print("y_tap : ",y_tap)
            
            # 280 + 
            # {'instances': Instances(num_instances=5, image_height=720, image_width=1280, 
            # fields=[pred_boxes: Boxes(tensor([[324.5747,   0.0000, 594.5106, 141.9667]

            # 324 - 594
            # 0.35673076923076924 * 1280 = 456.576
            
            # 0 - 141
            # ((1 - 0.7384615384615385) - 0.21875)/0.5625 *720

            preds = self.detections[0]
            # boundingBoxes = np.a
            scores =  self.detections[1]
            classes = self.detections[2]
            w,h = self.image_content.shape[:2]
            c1 = ((1280-720) / 2) / 1280
            c2 = (720) / 1280

            x_pixel, y_pixel = (((1-x_tap) - c1)/c2) * 720 , y_tap * 1280
            
            object_id = -1
            for id,coordinates in enumerate(preds):
                x1,y1,x2,y2 = coordinates
                
                if y1 <= x_pixel <= y2 and x1 <= y_pixel <= x2:
                    object_id = classes[id]
                    break
                
            detections = self.detections
            rospy.wait_for_service('/grasp_object/grasp_service_call')
            try:
                grasp_client = rospy.ServiceProxy('/grasp_object/grasp_service_call', grasp_service)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)
                return
            object_id = 10
            ret_val = grasp_client(object_id)

            sendFinishMessage("Done picking and placing the object!")

            return tap_serviceResponse(True) 
        except Exception as e:
            print(e)
            fail_message = "Could not pick the object"
            sendFinishMessage(fail_message)
            return tap_serviceResponse(True) 

    def image_update_callback(self, data):
        self.image_content = bridge.imgmsg_to_cv2(data)
        print("Updated image")
        self.sub.unregister()

    def service_callback(self, req):
        rospy.loginfo("Perception dropping service callback")
        # print("Image : ", type(image), image.shape)
        # image = bridge.imgmsg_to_cv2(image, "bgr8")

    def resize_img(self, im):
        desired_size = 1280
        old_size = im.shape[:2] # old_size is in (height, width) format
        
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        return new_im

    def annotate_image(self, image, annotations):
        # bp()
        bboxes = annotations[0]
        # bp()
        for bbox in bboxes:
            start_point = list(map(int, bbox[:2].tolist()))
            end_point = list(map(int, bbox[2:].tolist()))
            image = cv2.rectangle(image, start_point, end_point, (255,0,0), 2)
            
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        return image

    def trigger_response(self, request):
        try: 
            image = self.image_content
            # ============ Server to send the image for inference ============
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print("Attempting to connect")
            client_socket.connect((ip_address, 8485))
            print("connection Successful ")
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
            conn, addr = s.accept()
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
            
            outputs = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            self.detections = outputs
            image = self.annotate_image(image, outputs)
            image = self.resize_img(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_message = self.bridge.cv2_to_imgmsg(image, "passthrough")
            self.results_pub.publish(image_message)
            now = datetime.now() # current date and time

            date_time_str = now.strftime("%H:%M:%S")
            img_path = f"/home/hello-robot/trial/catkin_ws/src/stretch_ros/touri_ros/src/touri_perception/scripts/results/temp_img{date_time_str}.jpg"
            cv2.imwrite(img_path,image)
            time.sleep(0.1)
            uploadImg(img_path)
            # imageIter+=1
            return TriggerResponse(
                success=True,
                message="Updated image successfully!"
            )
        except Exception as e:
            print("Exception:", e)
            fail_message =  "Could not send the image to interface"
            sendFinishMessage(fail_message)
            return TriggerResponse(
                success=False,
                message=fail_message
            )

def main(args):
    rospy.init_node('send_img_service_node') 
    f = final()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    
if __name__ == '__main__':
    main(sys.argv)
