#!/usr/bin/env python3 
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel, Shivani"
__mail__      = "jkpatel@andrew.cmu.edu"
__copyright__ = "NONE"

# -----------------------------------------------------------------------------

from trimesh import transform_points
import roslib
import rospy
import math
import tf
import numpy as np
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
  
# -----------------------------------------------------------------------------
prev_point = None

def callback(msg,args):
    global prev_point
    try:
        trans,rot= listener.lookupTransform('/base_link','/link_head_tilt', rospy.Time(0))
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat   = tf.transformations.quaternion_matrix(rot)
        mat1 = np.dot(trans_mat, rot_mat)
        point_in_camera = np.array([msg.point.z, -msg.point.x, -msg.point.y, 1])
        point_in_base = np.matmul(mat1,point_in_camera)[0:3]
        if prev_point is None:
            prev_point = point_in_base
        else:
            if(np.linalg.norm(prev_point-point_in_base)>0.01):
                rospy.loginfo("Point in base:"+str(point_in_base))
                prev_point = point_in_base
        goal_point = PointStamped()
        goal_point.header.stamp = rospy.Time.now()
        goal_point.header.frame_id = '/base_link'
        goal_point.point.x = point_in_base[0]
        goal_point.point.y = point_in_base[1]
        goal_point.point.z = point_in_base[2]
        goal_pub.publish(goal_point)
            
    except Exception as e:
        print(e)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    rospy.init_node('touri_tf_listener')
    listener = tf.TransformListener()  
    goal_pub= rospy.Publisher("/goal_loc",PointStamped, queue_size=1)
    sub = rospy.Subscriber('/detected_point', PointStamped, callback,(listener))
    rospy.spin()