#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg
from std_msgs.msg import String
from touri_perception.msg import obj_det
import numpy as np
from pdb import set_trace as bp
import time
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
import os


# pub = rospy.Publisher('obs_det', obj_det, queue_size = 10)
# obstacle = obj_det()

def callback(msg):

    n = len(msg.ranges)
    window_size = 5

    # left 
    left_range = list(msg.ranges[n//8 : 3*n//8])
    left_range = np.array(left_range)
    left_range = left_range[np.where((left_range >= msg.range_min) & (left_range<= msg.range_max))]
    left_window_means = []

    if(left_range.shape[0] > window_size):

    # bp()
        for i in range(0,left_range.shape[0]-window_size,window_size):
            window = left_range[i:i+window_size]
            left_window_means.append(np.mean(window))

        left_window_min = min(left_window_means)


        if(msg.range_min <= left_window_min and left_window_min <= 0.4):
            l = 2
        
        elif(left_window_min > 0.4 and left_window_min <= 0.6):
            l = 1

        else:
            l = 0

    else:
        l = 0

    # backward
    backward_range = list(msg.ranges[3*n//8 : 5*n//8])
    backward_range = np.array(backward_range)
    
    backward_range = backward_range[np.where((backward_range >= msg.range_min) & (backward_range<= msg.range_max))]

    backward_window_means = []

    if(backward_range.shape[0] > window_size):

        for i in range(0,backward_range.shape[0]-window_size,window_size):
            window = backward_range[i:i+window_size]
            backward_window_means.append(np.mean(window))

        backward_window_min = min(backward_window_means)

        
        if(msg.range_min <= backward_window_min and backward_window_min <= 0.35):
            b = 2
        
        elif(backward_window_min > 0.35 and backward_window_min <= 0.6):
            b = 1

        else:
            b = 0

    else:
        b = 0

    # right
    right_range = list(msg.ranges[5*n//8 : 7*n//8])
    right_range = np.array(right_range)
    
    right_range = right_range[np.where((right_range >= msg.range_min) & (right_range<= msg.range_max))]

    right_window_means = []

    if(right_range.shape[0] > window_size):

        for i in range(0,right_range.shape[0]-window_size,window_size):
            window = right_range[i:i+window_size]
            right_window_means.append(np.mean(window))

        right_window_min = min(right_window_means)


        if(msg.range_min <= right_window_min and right_window_min <= 0.4):
            r = 2
        
        elif(right_window_min > 0.4 and right_window_min <= 0.6):
            r = 1

        else:
            r = 0

    else:
        r = 0

    # forward
    forward_range = list(msg.ranges[7*n//8 : n])
    forward_range+= list(msg.ranges[0 : n//8])
    forward_range = np.array(forward_range)

    forward_range = forward_range[np.where((forward_range >= msg.range_min) & (forward_range<= msg.range_max))]
    
    forward_window_means = []

    if(forward_range.shape[0] > window_size):

        for i in range(0,forward_range.shape[0]-window_size,window_size):
            window = forward_range[i:i+window_size]
            forward_window_means.append(np.mean(window))

        forward_window_min = min(forward_window_means)

        if(msg.range_min <= forward_window_min and forward_window_min <= 0.4):
            f = 2

        elif(forward_window_min > 0.4 and forward_window_min <= 0.6):
            f = 1

        else:
            f = 0

    else:
        f = 0

    updateProximity(f, b, l, r)

def updateProximity(top, bottom, left, right):
    '''
    OKAY => 0
    CLOSE => 1
    VERY CLOSE => 2
    '''
    assert top in (0,1,2), "Front proximity can only take values of 0,1,2"
    assert bottom in (0,1,2), "Front proximity can only take values of 0,1,2"
    assert left in (0,1,2), "Front proximity can only take values of 0,1,2"
    assert right in (0,1,2), "Front proximity can only take values of 0,1,2"
    db.reference("state/navObjProximity").update({
        "top" : top,
        "bottom": bottom,
        "left": left,
        "right": right,
    })


def listener():
    rospy.init_node('obstacle_alert', anonymous=True)
    sub = rospy.Subscriber('/scan_filtered', LaserScan, callback)
    rospy.spin()

if __name__ == '__main__':
    cred = credentials.Certificate("keys/touri-65f07-firebase-adminsdk-wuv71-b245c875f8.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/',
        'storageBucket' : 'touri-65f07.appspot.com' 
        })
    listener()
