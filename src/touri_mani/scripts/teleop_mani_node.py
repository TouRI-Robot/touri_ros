#!/usr/bin/env python
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel"
__mail__      = "jkpatel@andrew.cmu.edu"
__copyright__ = "NONE"

# -----------------------------------------------------------------------------

import rospy
from touri_mani.msg import TeleopMani
# from touri_core.touri_instance import touri_bot
from touri_core import touri_instance
import stretch_body.robot
import time

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def callback(data):
    rospy.loginfo("H: {}    E: {}".format(data.height, data.extend))
    
    lift_step = 0 if (data.height<0.2 and data.height>-0.2) else \
                    0.1*data.height
    arm_step  = 0 if (data.extend<0.2 and data.extend>-0.2) else \
                    (-0.1)*data.extend
    print("lift_step",lift_step)
    print("arm_step",arm_step)

    # touri_bot.lift.move_by(lift_step)
    # touri_bot.arm.move_by(arm_step)
    # touri_bot.push_command()

# -----------------------------------------------------------------------------

def listener():
    rospy.init_node('teleop_mani', anonymous=True)
    rospy.Subscriber("teleop_mani", TeleopMani, callback)
    rospy.spin()

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("TouRI BOT : ",touri_instance.touri_bot)
    # listener()
    # touri_bot._stop()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------