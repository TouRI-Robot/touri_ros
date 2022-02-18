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

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
objectron = mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=1,
                            min_detection_confidence=0.2,
                            min_tracking_confidence=0.99,
                            model_name='Cup')

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    
  def callback(self,data):
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
      if results.detected_objects:
          for detected_object in results.detected_objects:
              mp_drawing.draw_landmarks(
                image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
              mp_drawing.draw_axis(image, detected_object.rotation,
                                  detected_object.translation)
      # Flip the image horizontally for a selfie-view display.
      # cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
      # if cv2.waitKey(5) & 0xFF == 27:
      #   break
    except Exception as e:
      print(e)

    cv2.imshow("Image window", cv2.flip(image, 1))
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