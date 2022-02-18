#!/usr/bin/env python
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel"
__mail__      = "jkpatel@andrew.cmu.edu"
__copyright__ = "NONE"

# -----------------------------------------------------------------------------

import rospy
from touri_nav.msg import TeleopNav
from touri_core import touri_instance
import stretch_body.robot
import time

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def callback(data):
    print("TouRI BOT : ",str(touri_bot))
    
    rospy.loginfo("X: {}    Y: {}".format(data.x, data.y))
    trans_vel = 0 if (data.y<0.2 and data.y>-0.2) else 0.1*data.y
    rot_vel   = 0 if (data.x<0.2 and data.x>-0.2) else 0.3*data.x

    print("trans_vel",trans_vel)
    print("rot_vel",rot_vel)

    touri_bot.base.set_velocity(trans_vel,rot_vel)
    touri_bot.push_command()

# -----------------------------------------------------------------------------

def listener():
    rospy.init_node('teleop_nav', anonymous=True)
    rospy.Subscriber("teleop_nav", TeleopNav, callback)
    rospy.spin()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("TouRI BOT : ",touri_instance.touri_bot)
    listener()
    touri_bot._stop()
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------