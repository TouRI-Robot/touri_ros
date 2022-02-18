import rospy
from std srvs_srv import Trigger,TriggerRequest
from sensors_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

import hello_helpers.hello_misc as hm
import math

class Manipulation_t1(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)

        self.rate = 7.0

        self.random_pose1 = {'wrist_extension':0.3, 'joint_lift':0.5}
        self.random_pose2 = {'wrist_extension':0.25, 'joint_lift':0.5}

    def command_random(self):
        self.move_to_pose(self.random_pose1)
        self.trajectory_client.wait_for_result()
        self.move_to_pose(self.random_pose2)
        self.trajectory_client.wait_for_result()

    def begin_test(self):
        rate = rospy.Rate(self.rate)
        self.command_random()

    def main(self):
        hm.HelloNode.main(self,'manipulation_test1','manipulation_test1', wait_for_first_pointcloud = False)


if __name__ == '__main__':
    try:
        node = Manipulation_t1()
        node.main()
        node.begin_test()

    except KeyboardInterrupt:
        rospy.loginfo(" Interrupted ")
