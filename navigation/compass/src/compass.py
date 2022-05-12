#!/usr/bin/python3

from math import atan2, degrees

import rospy
from sensor_msgs.msg import MagneticField
from std_msgs.msg import Float32

class Compass:
    def __init__(self):
        """simple compass that subscribes to a magnetic field and publishes heading. Assumes no tilt. 
        """
        
        # parameters
        magnetometer_topic = rospy.get_param("magnetometer_topic", default='/zed2/zed_node/imu/mag')
        heading_topic = rospy.get_param("heading_topic", default="heading_magnetic")
        
        # set up sub and pub
        self.compass_pub = rospy.Publisher(heading_topic, Float32, queue_size=1)
        self.compass_pub_deg = rospy.Publisher(heading_topic + "_deg", Float32, queue_size=1)
        self.magnetometer_sub = rospy.Subscriber(magnetometer_topic, MagneticField, self.magnetometer_cb)
        

    def magnetometer_cb(self, mag_msg):
        heading = atan2(mag_msg.magnetic_field.y, mag_msg.magnetic_field.x)
        self.compass_pub.publish(heading)
        self.compass_pub_deg.publish(degrees(heading))

if __name__ == "__main__":
    rospy.init_node("compass")
    compass = Compass()
    rospy.spin()
