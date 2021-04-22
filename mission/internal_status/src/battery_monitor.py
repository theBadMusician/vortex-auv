#!/usr/bin/env python

#python imports
import subprocess
import os
import serial

#ros imports
import rospy
from std_msgs.msg import Float32

class BatteryMonitor():
    def __init__(self):
        
        rospy.init_node("battery_monitor")

        # Settings
        self.path_to_powersense = rospy.get_param("/battery/logging/powersense_dev")
        self.path_to_voltage_meter = rospy.get_param("/battery/logging/path")
        self.interval = rospy.get_param("/battery/logging/interval")            # How often the battery level is checked and published
        self.critical_level = rospy.get_param("/battery/thresholds/critical")   # Minimum allowed value in millivolts
        self.warning_level = rospy.get_param("/battery/thresholds/warning")     # Warning threshold in millivolts

        # Power Sense device
        rospy.loginfo("Setting up usb read..")
        self.powersense_device = serial.Serial(self.path_to_powersense, 115200)
        self.powersense_device.reset_input_buffer()
        
        xavier_voltage, system_voltage = self.get_voltages()    # try reading input

        # Publisher
        self.xavier_battery_level_pub = rospy.Publisher("/auv/battery_level/xavier", Float32, queue_size=1)
        self.system_battery_level_pub = rospy.Publisher("/auv/battery_level/system", Float32, queue_size=1)

        rospy.loginfo("BatteryMonitor initialized")

    
    def spin(self):
        
        try:
            while not rospy.is_shutdown():

                xavier_voltage, system_voltage = self.get_voltages()
                if not xavier_voltage == 0: 
                    self.xavier_battery_level_pub.publish(xavier_voltage)
                if not system_voltage == 0:
                    self.system_battery_level_pub.publish(system_voltage)

                self.log_voltage(xavier_voltage, "xavier")
                self.log_voltage(system_voltage, "system")

                rospy.sleep(self.interval)

        finally:
            self.powersense_device.close()
            

    def get_voltages(self):
        # Record output from voltage meter command, decode from bytes object to string, convert from string to integer
        xavier_mV = int(subprocess.check_output(["cat", self.path_to_voltage_meter]).decode("utf-8"))
        xavier_voltage = xavier_mV / 1000.0

        system_voltage_str = self.powersense_device.readline()
        system_voltage = float(system_voltage_str[:-2]) # strip /r/n
        self.powersense_device.reset_input_buffer()     
        # readline only reads the top line, so make sure 
        # the buffer is not filled with old voltage readings
        # by resetting the input buffer

        return xavier_voltage, system_voltage

    def log_voltage(self, voltage, title):

        if voltage == 0:
            rospy.loginfo("Voltage is zero. Killswitch is probably off.")
        
        #Critical voltage level
        elif voltage <= self.critical_level:
            rospy.logerr("Critical %s voltage: %.3fV" % (title, voltage))
            
        # Warning voltage level
        elif voltage <= self.warning_level:
            rospy.logwarn("%s voltage: %.3fV" % (title, voltage))

        else:
            rospy.loginfo("%s voltage: %.3fV" % (title, voltage))
            


if __name__ == '__main__':

        bm = BatteryMonitor()
        bm.spin()
