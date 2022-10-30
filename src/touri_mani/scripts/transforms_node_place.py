#!/usr/bin/env python3
from trimesh import transform_points
import roslib
# roslib.load_manifest('learning_tf')
import rospy
import math
import tf
import numpy as np
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped

# -----------------------------------------------------------------------------

def callback(msg,args):
    try:
        rospy.loginfo("listening")
        trans,rot= listener.lookupTransform('/base_link','/link_head_tilt', rospy.Time(0))
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat   = tf.transformations.quaternion_matrix(rot)
        mat1 = np.dot(trans_mat, rot_mat)
        point_in_camera = np.array([msg.point.z, -msg.point.x, -msg.point.y, 1])
        point_in_base = np.matmul(mat1,point_in_camera)[0:3]
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
    print("started node") 
    goal_pub= rospy.Publisher("/goal_to_place",PointStamped, queue_size=1)
    sub = rospy.Subscriber('/place_loc', PointStamped, callback,(listener))
    rospy.spin()
