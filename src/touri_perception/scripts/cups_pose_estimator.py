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
import cv_bridge 
from cv_bridge import CvBridge, CvBridgeError
import time
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image

import message_filters
import mediapipe as mp

# -----------------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
objectron = mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=10,
                            min_detection_confidence=0.05,
                            min_tracking_confidence=0.1,
                            model_name='Cup')

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    image_sub = message_filters.Subscriber("/camera/color/image_raw",Image)
    depth_image_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_image_sub],1,5,allow_headerless=True)
    ts.registerCallback(self.callback)
    self.br = CvBridge()

  def callback(self,data,depth_image):
    try:
      image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
      image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
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
      
      depth_data = np.frombuffer(depth_image.data, dtype=np.uint16).reshape(depth_image.height, depth_image.width, -1)
      
      # print("depth_data : ",depth_data.shape)
      # print("image : ",image.shape)
      cups_position = []
      if results.detected_objects:
          for detected_object in results.detected_objects:
              mp_drawing.draw_landmarks(
                image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
              mp_drawing.draw_axis(image, detected_object.rotation,
                                  detected_object.translation)
              
              # x = []
              # y = []
              # for landmark in detected_object.landmarks_2d.landmark:
              #     x.append(landmark.x)
              #     y.append(landmark.y)

              # Y = int(720*np.mean(np.array(x)))
              # X = int(1080*np.mean(np.array(y)))
              
              # landmarks_3d = {}
              # default_z_3d = -1
      
              # # print(Y,X)
              # # print('depth shape : ',depth_data.shape)
              # Z = depth_data[Y-10:Y+10,X-10:X+10]
              # z = np.mean(Z[np.nonzero(Z)])
              # # print(z)

              # if z > 0: 
              #     z_3d = z / 1000.0
              #     x_3d = ((X - c_x) / f_x) * z_3d
              #     y_3d = ((Y - c_y) / f_y) * z_3d
              #     print(f"x_3d, y_3d, z_3d : {x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}")
              #     cups_position.append((x_3d, y_3d, z_3d))
              # else:
              #     z_3d = default_z_3d
    except Exception as e:
      print(e)

    self.image_pub.publish(self.br.cv2_to_imgmsg(image))
    # Challenge
    # cv2.imshow("Image window", image)
    # cv2.waitKey(1000)
    



def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)