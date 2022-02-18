#!/usr/bin/env python

import firebase_admin
from firebase_admin import credentials, db

import rospy
from touri_nav.msg import TeleopNav
from touri_mani.msg import TeleopMani
import time

#GLOBAL VARIABLES
rospy.init_node('user_command_listerner', anonymous=True)
nav_pub = rospy.Publisher('teleop_nav', TeleopNav)
mani_pub = rospy.Publisher('teleop_mani', TeleopMani)


def init_database():
    '''
    Init firebase database
    Uses a json key (not in the repo) to form a webSocket connection with the database
    '''
    cred = credentials.Certificate("/home/hello-robot/Desktop/TouRI/touri_ros/src/touri_core/keys/touri-65f07-firebase-adminsdk-wuv71-3751c21aa8.json")
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://touri-65f07-default-rtdb.firebaseio.com/'})


def udpate_db_vals():
    print("sending")
    db.reference("/nav").update({"x":0.5,"y":0.5})
    db.reference("/mani").update({"height":0.5,"extend":0.5})

if __name__=="__main__":
    init_database()
    while True:
        udpate_db_vals()