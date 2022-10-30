#!/usr/bin/env python3
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
import tf
from geometry_msgs.msg import PointStamped
import hello_helpers.hello_misc as hm


hm.HelloNode.main('manipulation', 'manipulation', wait_for_first_pointcloud=False)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TouRI_bot
class TouRI_bot(Robot):
    def __init__(self):
        Robot.__init__(self)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.startup()
        self.base.startup()
        self.lift.startup()
        self.arm.startup()
        self.head.startup()
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

touri_bot = TouRI_bot()

# -----------------------------------------------------------------------------
def callback_nav(data):
    rospy.loginfo("X: {}    Y: {}".format(data.x, data.y))
    trans_vel = 0 if (data.y<0.2 and data.y>-0.2) else 0.1*data.y
    rot_vel   = 0 if (data.x<0.2 and data.x>-0.2) else 0.3*data.x
    touri_bot.base.set_velocity(trans_vel,rot_vel)
    touri_bot.push_command()

# -----------------------------------------------------------------------------

def callback_mani(data):
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

def callback_goal(data):
    print("data.point.x",data.point.x)
    print("data.point.y",data.point.y)
    print("data.point.z",data.point.z)
    touri_bot.goal_x= data.point.x
    touri_bot.goal_y= data.point.y
    touri_bot.goal_z= data.point.z

# -----------------------------------------------------------------------------

def touri_actuate():
    rospy.init_node('touri_core',anonymous=True)
    rospy.Subscriber("teleop_nav", TeleopNav, callback_nav)
    rospy.Subscriber("teleop_mani", TeleopMani, callback_mani)
    rospy.Subscriber("goal",PointStamped, callback_goal)
    rospy.spin()

# -----------------------------------------------------------------------------
if __name__=="__main__":
    rospy.init_node('Manipulation_started')
    print("starting robot")
    sub = rospy.Subscriber("/goal",PointStamped, callback_goal)


    # open=100
    # touri_bot.arm.move_to(0.0)
    # touri_bot.end_of_arm.move_to('stretch_gripper',open)
    # touri_bot.head.move_to('head_pan', 0.0)
    # touri_bot.head.move_to('head_tilt', 0.0)
    # touri_bot.push_command()
    
    # time.sleep(10)
    # print("step 0 done")
    
    # touri_bot.lift.move_to(0.15)
    # time.sleep(10)
    # # Step 1 :
    # #X = 0.4
    # X= touri_bot.goal_x - 0.8
    # touri_bot.base.translate_by(X)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("Translate by:",X)
    # print("step 1 done")
    
    # # Step 2 :
    # theta = 1.57
    # touri_bot.base.rotate_by(theta)
    # touri_bot.push_command()
    # print("Rotate by:", theta)
    # print("step 2 done")

    # time.sleep(10)
    
    # # Step 3 :
    # Y = touri_bot.goal_y
    # touri_bot.base.translate_by(Y)
    # touri_bot.push_command()

    # time.sleep(10)
    # print("Translate by:", Y)
    # print("step 3 done")
    # x_move_head_pan = -1.57
    # z_move_head_pan = -1.57/3
    # touri_bot.head.move_by('head_pan', x_move_head_pan)
    # touri_bot.head.move_by('head_tilt', z_move_head_pan)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("Touri robot sleep")
    
    # Step 4 :
    #visual feedback based
    # V = 0.05
    # touri_bot.base.translate_by(V)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("step 4 done")
    # #lift_step = 0.8
    # lift_step= touri_bot.goal_z
    # arm_step = 0.4
    # #open= 60
    # close = 0
    # grasp_lift= 0.9
    # touri_bot.lift.move_to(lift_step)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("Lift by:",lift_step )
    # print("step 5 done")
    
    # touri_bot.arm.move_to(arm_step)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("Extend by:",arm_step)
    # print("step 6 done")

    # touri_bot.end_of_arm.move_to('stretch_gripper',close)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("close by:",close)
    # print("step 8 done")

    # touri_bot.lift.move_to(grasp_lift)
    # touri_bot.push_command()
    # time.sleep(10)
    # print("Lift by:",grasp_lift)
    # print("step 9 done")

    # touri_bot.base.set_velocity(trans_vel,rot_vel)
    # touri_bot.base.rotate_by(1.57)
    # lift_step = 0.1
    # arm_step = 0.1
    # touri_bot.lift.move_by(lift_step)
    # touri_bot.arm.move_by(arm_step)

    # touri_bot.lift .move_to(0.2)
    # touri_bot.arm.move_to(0.0)
    # touri_bot.head.home()
    # starting_position = touri_bot.head.status['head_pan']['pos']
    # print("starting_position : ",starting_position)
    # rospy.loginfo("lifting and extending")

    # touri_bot.push_command()
    
    # look right by 90 degrees
    # touri_bot.head.move_to('head_pan', starting_position + 1.57)
    # touri_bot.head.get_joint('head_pan').wait_until_at_setpoint()

    # # tilt up by 30 degrees
    # 'head_pan' or 'head_tilt'
    # Head Pose 1 -  

    # touri_bot.head.move_to('head_pan', 0)
    # touri_bot.head.get_joint('head_pan').wait_until_at_setpoint()

    # touri_bot.head.move_to('head_tilt', (0))
    # touri_bot.head.get_joint('head_tilt').wait_until_at_setpoint()

    # Head Pose 2 -  
    # touri_bot.head.move_to('head_pan', -1.57)
    # touri_bot.head.get_joint('head_pan').wait_until_at_setpoint()

    # touri_bot.head.move_to('head_tilt', (-1.57/3))
    # touri_bot.head.get_joint('head_tilt').wait_until_at_setpoint()

    # # # look down towards the wheels
    # touri_bot.head.move_to('head_pan', 0)
    # touri_bot.head.get_joint('head_pan').wait_until_at_setpoint()

    # Head Pose 3 -      

    # touri_bot.head.move_to('head_pan', -1.57)
    # touri_bot.head.get_joint('head_pan').wait_until_at_setpoint()
    # touri_bot.head.move_to('head_tilt', (-1.57))
    # touri_bot.head.get_joint('head_tilt').wait_until_at_setpoint()
    
    # touri_bot.head.pose('wheels')

    # import time; time.sleep(3)

    # look ahead
    # touri_bot.head.pose('ahead')
    # time.sleep(3)

    # rospy.loginfo("====done")
    
    # rospy.spin()
    # touri_actuate()

