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
import mediapipe as mp
# from __future__ import print_function
import sys

# import roslib
# roslib.load_manifest('my_package')
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    
  def callback(self,data):
    try:
      cv_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    except Exception as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)



def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)