#!/usr/bin/env python

import firebase_admin
from firebase_admin import credentials, db

import rospy
from touri_nav.msg import TeleopNav
from touri_mani.msg import TeleopMani

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


def nav_listener(event):
    '''
    Event listener that reacts to changes in event db events for navigation values
    1. Gets raw values from the database i.e. joystick X & Y
    2. Creates a 'TeleopNav' msg and publishes it onto nav_pub's topic
    NOTE: Function expects entire 'nav' db-node to be updated
    '''
    rospy.loginfo("FROM FIREBASE (NAV) -> X: {}    Y:{}".format(event.data['x'], event.data['y']))
    msg = TeleopNav()
    msg.x = event.data['x']
    msg.y = event.data['y']
    nav_pub.publish(msg)
    

def mani_listner(event):
    '''
    Event listener that reacts to changes in event db events for navigation values
    1. Gets raw values from the database i.e. joystick X & Y
    2. Creates a 'TeleopMani' msg and publishes it onto mani_pub's topic
    NOTE: Function expects entire 'mani' db-node to be updated
    '''
    rospy.loginfo("FROM FIREBASE (MANI) -> H: {}    E:{}".format(event.data['height'], event.data['extend']))
    msg = TeleopMani()
    msg.height = event.data['height']
    msg.extend = event.data['extend']
    mani_pub.publish(msg)

def main():
    init_database()
    #Add listners to DB channels
    db.reference("/nav").listen(nav_listener)
    db.reference("/mani").listen(mani_listner)


if __name__ == "__main__":
    try:
        main()
        # touri.stop()
    except KeyboardInterrupt:
        pass
    
