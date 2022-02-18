#!/usr/bin/env python
"""
TouRI Robot Base Code
"""
__author__    = "Jigar Patel"
__mail__      = "jkpatel@andrew.cmu.edu"
__copyright__ = "NONE"

# -----------------------------------------------------------------------------

import rospy
from stretch_body.robot import Robot
from touri_mani.msg import TeleopMani
from touri_nav.msg import TeleopNav
from pynput import keyboard
import time

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TouRI_bot
class TouRI_bot(Robot):
    def __init__(self):
        Robot.__init__(self)
        self.startup()
        self.base.startup()
        self.lift.startup()
        self.arm.startup()
        self.setup_motion_vars()
    
    # -------------------------------------------------------------------------

    def setup_motion_vars(self):
        self.base_trans_vel = 0
        self.base_rot_vel   = 0
        self.lift_step   = 0.1
        self.arm_step   = 0.1
    
    # -------------------------------------------------------------------------

    def _stop(self):
        self.arm.stop()
        self.lift.stop()
        self.base.stop()
        self.stop()

    # -------------------------------------------------------------------------

    def __str__(self):
        return "TouRI - An RI Touring Robot"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TouRI
class TouRI():
    _instance = None

    def __init__(self):
        raise RuntimeError('Call TouRI.instance() instead')

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = TouRI_bot()
        return cls._instance

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
touri_bot = TouRI.instance()


# -----------------------------------------------------------------------------
def callback_nav(data):
    rospy.loginfo("X: {}    Y: {}".format(data.x, data.y))
    trans_vel = 0 if (data.y<0.2 and data.y>-0.2) else 0.1*data.y
    rot_vel   = 0 if (data.x<0.2 and data.x>-0.2) else 0.3*data.x
    touri_bot.base.set_velocity(trans_vel,rot_vel)
    touri_bot.push_command()

# -----------------------------------------------------------------------------

def callback_mani(data):
    rospy.loginfo("H: {}    E: {}".format(data.height, data.extend))
    lift_step = 0 if (data.height<0.2 and data.height>-0.2) else \
                    0.1*data.height
    arm_step  = 0 if (data.extend<0.2 and data.extend>-0.2) else \
                    (-0.1)*data.extend
    print("lift_step",lift_step)
    print("arm_step",arm_step)
    touri_bot.lift.move_by(lift_step)
    touri_bot.arm.move_by(arm_step)
    touri_bot.push_command()

# -----------------------------------------------------------------------------

def touri_actuate():
    rospy.init_node('touri_core',anonymous=True)
    rospy.Subscriber("teleop_nav", TeleopNav, callback_nav)
    rospy.Subscriber("teleop_mani", TeleopMani, callback_mani)
    rospy.spin()

# -----------------------------------------------------------------------------
if __name__=="__main__":
    print(touri_bot)
    touri_actuate()

