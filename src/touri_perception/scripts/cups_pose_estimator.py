#!/usr/bin/env python3
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel"
__mail__      = "jkpatel@andrew.cmu.edu"
__copyright__ = "NONE"

# -----------------------------------------------------------------------------

import cv2
import numpy as np
import rospy
import cv_bridge 
from cv_bridge import CvBridge, CvBridgeError
import time
import sys
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Pose, Quaternion, Twist, Vector3
from std_msgs.msg import String, Bool
import message_filters
import mediapipe as mp
from touri_perception.srv import transform_service,transform_serviceResponse
from std_srvs.srv import Trigger, TriggerRequest
from touri_perception.srv import perception,perceptionResponse

# -----------------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
objectron = mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=1,
                            min_detection_confidence=0.15,
                            # min_tracking_confidence=0.2,
                            model_name='Cup')

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.depth_pub = rospy.Publisher("depth_topic_2",Image)
    self.flag_pub= rospy.Publisher("/flag",Bool, queue_size=10)
    image_sub = message_filters.Subscriber("/camera/color/image_raw",Image)
    depth_image_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_image_sub],1,5,allow_headerless=True)
    ts.registerCallback(self.callback)
    self.br = CvBridge()
    self.image_content = None
    self.depth_content = None
    self.pose_pub = rospy.Publisher('/detected_point', PointStamped, queue_size=10)
    self.flag = False
    self.image_server = rospy.Service('perception_service', perception, self.service_callback)
    
  def callback(self,data,depth_image):
    self.image_content = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    self.depth_content = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(depth_image.height, depth_image.width, -1)
      
  def service_callback(self,req):
    try:
      print("service callabck")
      image = cv2.rotate(self.image_content, cv2.cv2.ROTATE_90_CLOCKWISE)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = objectron.process(image)
      
      
      # Draw the box landmarks on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      Y,X = image.shape[:2]

      # camera_matrix = np.array([[640.6170043945312, 0.0, 645.9562377929688], [0.0, 640.6170043945312, 357.26904296875], [0.0, 0.0, 1.0]])
      camera_matrix = np.array([[910.7079467773438, 0.0, 634.5316772460938], [0.0, 910.6213989257812, 355.40097045898], [0.0, 0.0, 1.0]])
      
      
      f_x = camera_matrix[0,0]
      c_x = camera_matrix[0,2]
      f_y = camera_matrix[1,1]
      c_y = camera_matrix[1,2]
      
      depth_data = self.depth_content
      Y = 0
      X = 0
      cups_position = []
      detected = False
      if results.detected_objects:
          for detected_object in results.detected_objects:
              # mp_drawing.draw_landmarks(
              #   image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
              # mp_drawing.draw_axis(image, detected_object.rotation,
              #                     detected_object.translation)
              
              x = []
              y = []
              for landmark in detected_object.landmarks_2d.landmark:
                  x.append(landmark.x)
                  y.append(landmark.y)
              
              Y = int(720*np.mean(np.array(x)))
              X = int(1280*np.mean(np.array(y)))
              
              Y = 720 - Y
              
              landmarks_3d = {}
              default_z_3d = -1

              Z = depth_data[Y-10:Y+10,X-10:X+10]

              image = cv2.rectangle(image, (720-Y-30,X-30), (720 - Y+30,X+30), (255,0,0), 20)
              image = cv2.circle(image,(720-Y,X), 10, (0,0,255), -1)
              z = np.mean(Z[np.nonzero(Z)])
              if z!=None:
                self.flag= True

              if z > 0: 
                  z_3d = z / 1000.0
                  x_3d = ((X - c_x) / f_x) * z_3d
                  y_3d = ((Y - c_y) / f_y) * z_3d
                  point  = PointStamped()
                  point.header.stamp = rospy.Time.now()
                  point.header.frame_id = '/map'
                  point.point.x = x_3d
                  point.point.y = y_3d
                  point.point.z = z_3d
                  
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
                  # self.pose_pub.publish(point)
                  self.image_pub.publish(self.br.cv2_to_imgmsg(image))
                  self.depth_pub.publish(self.br.cv2_to_imgmsg(depth_data))
                  detected = True
                  return perceptionResponse(detected,x_base,y_base,z_base)

                  cups_position.append((x_3d, y_3d, z_3d))
              else:
                  z_3d = default_z_3d
      else:
        detected = False
        # self.flag = False
        return perceptionResponse(detected,-1,-1,-1)
      
      b = Bool()
      b.data = self.flag
      self.flag_pub.publish(b)

    except Exception as e:
      print(e)

    # self.image_pub.publish(self.br.cv2_to_imgmsg(image))
    # self.depth_pub.publish(self.br.cv2_to_imgmsg(depth_data))
    
def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    rospy.loginfo("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)