#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
import sensor_msgs.msg
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped  
import numpy as np
from pdb import set_trace as bp
import time
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
import os
import threading


SAMPLING_TIME = 1

time_at_last_update = 0

room = "None"

def sendFinishMessage(msg: str):
    try : 
        cred = credentials.Certificate("keys/touri-65f07-firebase-adminsdk-wuv71-b245c875f8.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/',
            'storageBucket' : 'touri-65f07.appspot.com' 
        })
    except:
        pass  
    
    db.reference("state").update({
        "botMsg" : msg,
        "runningAutoCommand" : False,
    })



def callback(msg):
    global time_at_last_update

    pos_x = msg.pose.pose.position.x
    pos_y = msg.pose.pose.position.y

    threshold = 0.15

    roomData = {
        "ai_makerspace" : (26.1, 27.7, 0.92, 0.39),
        "gym": (37.84, 23.03, -0.37, -0.92),
        "start" : (0,0,0,0)
    }

    if room != 'None':
        x , y, z, w = roomData[room]
        if (x - threshold < pos_x < x + threshold) and (y - threshold < pos_y < y + threshold):
            sendFinishMessage("Reached location")
            db.reference("autoSkills/navigation/").update({'selectedRoom' : "None"})
            
        print("CURRENT X: {}     Y: {}".format(pos_x, pos_y))


        if (time.time() - time_at_last_update > SAMPLING_TIME):
            # print("UPDATING LOCATION")
            updateBotPosInMap(pos_x, pos_y)
            time_at_last_update = time.time()

# ---------------------------------------------------------------------------- #

def updateBotPosInMap(x,y):

    try : 
        cred = credentials.Certificate("keys/touri-65f07-firebase-adminsdk-wuv71-b245c875f8.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/',
            'storageBucket' : 'touri-65f07.appspot.com' 
        })
    except:
        pass  

    scale = 0.0255275

    x = x * scale
    y = y * scale
    x = x + 0.00268
    y =  0.8035 - y

    assert abs(x) <= 1, "X must be between 0 and 1"
    assert abs(y) <= 1, "Y must be between 0 and 1"
    db.reference("autoSkills/navigation/botPos").update({
        "x" : x,
        "y": y
    })


# ---------------------------------------------------------------------------- #


def getCurrentGoalLocation(event):
    global room
    room = event.data





# ---------------------------------------------------------------------------- #
def listener():
    rospy.init_node('pos_update', anonymous=True)
    sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    cred = credentials.Certificate("/home/hello-robot/trial/touri_integration/keys/touri-65f07-firebase-adminsdk-wuv71-b245c875f8.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/',
        'storageBucket' : 'touri-65f07.appspot.com' 
        })
    db.reference("autoSkills/navigation/selectedRoom").listen(getCurrentGoalLocation)
    listener()
