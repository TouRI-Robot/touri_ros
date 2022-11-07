#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg
from std_msgs.msg import String
# from obs_det.msg import obj_det

# pub = rospy.Publisher('obs_det', obj_det, queue_size = 10)
# obstacle = obj_det()
# -rw-rw-r--  1 hello-robot hello-robot 231K Oct  9 15:38  camera_image.jpeg
# -rw-rw-r--  1 hello-robot hello-robot 8.9K Oct 10 16:18  centroid_calc.py
# -rw-rw-r--  1 hello-robot hello-robot   35 Oct  5 11:19 'check queue size in navigaiton.rviz'
# -rw-rw-r--  1 hello-robot hello-robot 6.1K Oct 29 17:04  collect_training_data.py
# -rw-rw-r--  1 hello-robot hello-robot 5.9K Sep 18 14:28  collect_training_data_raw.py
# -rwxrwxr-x  1 hello-robot hello-robot 5.2K Apr 19  2022  cups_pose_estimator_backup.py
# -rwxrwxr-x  1 hello-robot hello-robot 6.5K Oct 29 19:08  cups_pose_estimator.py
# -rw-rw-r--  1 hello-robot hello-robot 5.3K Apr 14  2022  cups_pose_estimator.pyc
# -rwxrwxr-x  1 hello-robot hello-robot 6.8K Apr 18  2022  detect_plane.py
# -rwxrwxr-x  1 hello-robot hello-robot 6.8K Oct  5 11:11  dropping_plane.py
# -rwxrwxr-x  1 hello-robot hello-robot  21K Oct 29 21:01  final_centroid_calc.py
# -rwxrwxr-x  1 hello-robot hello-robot  14K Oct 26 16:13  final_centroid_convert.py
# -rwxrwxr-x  1 hello-robot hello-robot  18K Oct 11 11:19  final.py
# -rwxrwxr-x  1 hello-robot hello-robot  734 Oct 28 13:23  lidar_sub.py
# -rwxrwxr-x  1 hello-robot hello-robot 3.6K Oct 28 22:42  obstacle_alert.py
# -rw-rw-r--  1 hello-robot hello-robot 2.2K Oct  9 16:05  ransac.py
# -rw-rw-r--  1 hello-robot hello-robot  21K Oct 29 21:03  shipping_box_detection.py
# -rwxrwxr-x  1 hello-robot hello-robot  10K Oct 29 19:48  souvenir_pose_estimator.py
# -rw-rw-r--  1 hello-robot hello-robot 6.4K Oct 10 16:07  touri_depth_est_sh.py
# -rwxrwxr-x  1 hello-robot hello-robot 7.7K Oct 10 16:11  training_inference_all.py
# -rwxrwxr-x  1 hello-robot hello-robot 7.6K Oct 10 16:47  training_inference.py
# -rwxrwxr-x  1 hello-robot hello-robot 4.2K Sep 23 15:54  training.py
# -rwxrwxrwx  1 hello-robot hello-robot  825 Oct 11 01:34  transform_client.py
# -rwxrwxrwx  1 hello-robot hello-robot 1.8K Oct 28 15:09  transform_server.py

def callback(msg):

    print(len(msg.ranges))
    lidar_length = ((msg.angle_max - msg.angle_min)/msg.angle_increment)
    print("angle max: ",msg.angle_max)
    print("angle min: ",msg.angle_min)
    print("angle increment: ",msg.angle_increment)
    
    print("lidar range length:", lidar_length)

def listener():
    rospy.init_node('read_scan', anonymous=True)
    sub = rospy.Subscriber('/scan', LaserScan, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()