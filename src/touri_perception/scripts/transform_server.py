#!/usr/bin/env python3

from __future__ import print_function

from touri_perception.srv import transform_service,transform_serviceResponse
import rospy
import tf
import numpy as np

def transform_to_base_fun(req):
    print("Returning [%s , %s , %s]"%((req.x_request+1), (req.y_request+1),(req.z_request + 1)))
    rospy.loginfo("listening")
    trans,rot= listener.lookupTransform('/base_link','/link_head_tilt', rospy.Time(0))
    trans_mat = tf.transformations.translation_matrix(trans)
    rot_mat   = tf.transformations.quaternion_matrix(rot)
    mat1 = np.dot(trans_mat, rot_mat)
    # point_in_camera = np.array([msg.point.z, -msg.point.x, -msg.point.y, 1])
    point_in_camera = np.array([req.z_request, -req.x_request, -req.y_request, 1])
    point_in_base = np.matmul(mat1,point_in_camera)[0:3]
    print("Point in base:", point_in_base)
    # goal_point = PointStamped()
    # goal_point.header.stamp = rospy.Time.now()
    # goal_point.header.frame_id = '/base_link'
    # goal_point.point.x = point_in_base[0]
    # goal_point.point.y = point_in_base[1]
    # goal_point.point.z = point_in_base[2]
    # goal_pub.publish(goal_point)
    reply_x = point_in_base[0]
    reply_y = point_in_base[1]
    reply_z = point_in_base[2]
    return transform_serviceResponse(reply_x,reply_y,reply_z)

if __name__ == "__main__":
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('transform_to_base', transform_service, transform_to_base_fun)
    print("Ready to transform to base node")
    listener = tf.TransformListener() 
    # goal_pub= rospy.Publisher("/goal_loc",PointStamped, queue_size=1)
    # sub = rospy.Subscriber('/detected_point', PointStamped, callback,(listener))
    rospy.spin()
    
    
# rosservice call /add_two_ints "x_request:4
# y_request:3
# z_request:40" 

