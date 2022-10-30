#!/usr/bin/env python3

from __future__ import print_function

import sys
import rospy
from touri_perception.srv import transform_service,transform_serviceResponse

def add_two_ints_client(x, y, z):
    rospy.wait_for_service('transform_to_base')
    try:
        add_two_ints = rospy.ServiceProxy('transform_to_base', transform_service)
        resp1 = add_two_ints(x, y, z)
        print(resp1.x_reply,resp1.y_reply,resp1.z_reply)
        print("Ready to transform to base node")
        return resp1
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    x,y,z = 1,2,3
    print("Requesting %s+%s"%(x, y))
    response = add_two_ints_client(x, y, z)
    # print("%s + %s = %s"%(x, y, add_two_ints_client(x, y, z)))