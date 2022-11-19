#!/usr/bin/env python3

from __future__ import print_function

from touri_perception.srv import transform_service,transform_serviceResponse
import rospy
import tf
import numpy as np

def transform_to_base_fun(req):
    print("Returning [%s , %s , %s]"%((req.x_request), (req.y_request),(req.z_request)))
    rospy.loginfo("listening")
    # link_head_tilt
    trans, rot= listener.lookupTransform('/base_link','/link_head_tilt', rospy.Time(0))
    trans_mat = tf.transformations.translation_matrix(trans)
    rot_mat   = tf.transformations.quaternion_matrix(rot)
    mat1 = np.dot(trans_mat, rot_mat)
    
    print('trans_mat : ',trans_mat)
    print('rot_mat : ',rot_mat)
    
    point_in_camera = np.array([req.z_request, -req.x_request, -req.y_request, 1])
    trans_mat = np.array([[ 1. , 0. , 0., -0.01072437],
                          [ 0. , 1. , 0.,  0.01620489],
                          [ 0. , 0. , 1.,  1.29963565],
                          [ 0. , 0. , 0.,  1.        ]])
                          
    rot_mat = np.array([[ 1  , 0  ,  0, 0. ],
                        [ 0  , 0  , -1, 0. ],
                        [ 0  , 1  ,  0, 0. ],
                        [ 0. , 0. ,  0, 1. ]])

    rot_mat = np.array([[ 0.866 , 0.50  , 0.0 , 0.  ],
                          [ 0.0   , 0.0   , -1  , 0.  ],
                          [-0.50  , 0.866 ,  0  , 0.  ],
                          [ 0.    , 0.    ,  0. , 1.  ]])
    rot_mat2 = np.array([[ 0.0         ,0.0       , -1.0 , 0.  ],
     [-0.866       ,-0.50     , -0.0 , 0.  ],
     [-0.55685491  ,0.83054287,  0.0 , 0.  ],
     [ 0.          ,0.        ,  0.  , 1.  ]])
     
    mat1 = np.dot(trans_mat, rot_mat)
    print('point_in_camera : ',point_in_camera)
    point_in_base = np.matmul(mat1,point_in_camera)[0:3]

    reply_x = point_in_base[0]
    reply_y = point_in_base[1]
    reply_z = point_in_base[2]
    return transform_serviceResponse(reply_x,reply_y,reply_z)

if __name__ == "__main__":
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('transform_to_base', transform_service, transform_to_base_fun)
    print("Ready to transform to base node")
    listener = tf.TransformListener() 
    rospy.spin()
    
