#!/usr/bin/env python
# coding: UTF-8

"""

Node som forenkler tilganger til controlleren. 
Den skal ta av seg bytting mellom controller moduser, 
slik at koden i state machinene kan bli enklere. 

Noden skal ha en action-server som tar inn en ny action (ligner på
move_base) som skal inneholde: 
- Target pose og 
- String for kontroller som skal benyttes

Noden sender så target posen videre som en ny action
til den valgte kontrolleren. Resultat og feedback fra endelig kontroller
sendes videre igjennom noden. 

"""


import rospy
import actionlib
import time

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from vortex_msgs.msg import (
    MoveAction, MoveActionFeedback, MoveActionResult, LosPathFollowingAction,
    LosPathFollowingGoal, ControlModeAction
)
from vortex_msgs.srv import ControlMode


# ENUM for controller mode
OPEN_LOOP           = 0
POSE_HOLD           = 1
HEADING_HOLD        = 2
DEPTH_HEADING_HOLD  = 3
DEPTH_HOLD          = 4
POSE_HEADING_HOLD   = 5
CONTROL_MODE_END    = 6


def change_control_mode(requested_mode):
    """
    Change the controller mode of the DP controller.

    Although this is technically breaking the design
    idea that FSM and controller should not talk to
    eachother, it makes sense to keep it for now, seeing
    as some guidance systems may rely on one of the control
    modes for the controller.

    In the future, it may be moved to the DP guidance node,
    if it makes sense to.

    """

    rospy.wait_for_service('/controller/controlmode_service')   # From controller_ros.cpp

    try:
        control_mode = rospy.ServiceProxy('/controller/controlmode_service', ControlMode)
        response = control_mode(requested_mode)
        return response.result

    except rospy.ServiceException as e:
        rospy.logerr('guidance_interface could not change control mode')
        print('Service call failed: %s' %e)


class GuidanceInterface:

    def __init__(self):
        """
        Define constants used in the guidance systems, create the
        move action server that the fsm uses to communicate with 
        this node,  and connect to the actions servers in the guidance
        systems.
        """

        self.transit_speed = rospy.get_param('~transit_speed', 0.3)
        self.sphere_of_acceptance = rospy.get_param('~sphere_of_acceptance', 0.5)
        self.timeout = rospy.get_param('~guidance_interface_timeout', 90)

        self.action_server = actionlib.SimpleActionServer('move', MoveAction, self.move_cb, auto_start=False)
        self.action_server.start()

        self.dp_client = actionlib.SimpleActionClient('dp_action_server', MoveBaseAction)
        self.los_client = actionlib.SimpleActionClient('los_action_server', LosPathFollowingAction)

        self.joystick_controlmode_server = actionlib.SimpleActionServer('control_mode_server',
            ControlModeAction, self.joystick_control_mode_cb, auto_start=False)
        self.joystick_controlmode_server.start()

        rospy.loginfo('Guidance interface is up and running')


    def move_cb(self, move_goal):
        """
        Converts move_goal into the proper goal type for the desired guidance
        system and engages the corresponding guidance node through the
        action servers.

        Aborts action if it is not completed within self.timeout seconds
        """

        # Talk to the dp_guidance node...
        if move_goal.guidance_type == 'PositionHold':
            rospy.loginfo('move_cb -> PositionHold. Changing control mode...')
            change_control_mode(POSE_HEADING_HOLD)

            dp_goal = MoveBaseGoal()
            dp_goal.target_pose.pose = move_goal.target_pose

            self.dp_client.send_goal(dp_goal, done_cb=self.done_cb, feedback_cb=None)

            if not self.dp_client.wait_for_result(timeout=rospy.Duration(self.timeout)):
                self.action_server.set_aborted()
                rospy.loginfo('DP guidance aborted action due to timeout')


        # Talk to the los_guidance node...
        elif move_goal.guidance_type == 'LOS':
            rospy.loginfo('move_cb -> LOS. Changing control mode...')
            change_control_mode(OPEN_LOOP)

            los_goal = LosPathFollowingGoal()
            los_goal.next_waypoint = move_goal.target_pose.position
            los_goal.forward_speed.linear.x = self.transit_speed
            los_goal.sphereOfAcceptance = self.sphere_of_acceptance
            los_goal.desired_depth.z = move_goal.target_pose.position.z

            self.los_client.send_goal(los_goal, done_cb=self.done_cb, feedback_cb=None)

            if not self.los_client.wait_for_result(timeout=rospy.Duration(self.timeout)):
                self.action_server.set_aborted()
                rospy.loginfo('LOS guidance aborted action due to timeout')
        
        else:
            rospy.logerr('Unknown guidace type sent to guidance_interface')
            self.action_server.set_aborted()

    
    def done_cb(self, state, result):
        """
        Set the outcome of the action depending on
        the returning result.
        """
        
        if state == GoalStatus.SUCCEEDED:
            self.action_server.set_succeeded()

        elif state == GoalStatus.PREEMPTED:
            self.action_server.set_preempted()

        else:
            self.action_server.set_aborted()

    def joystick_control_mode_cb(self, control_mode):
        change_control_mode(control_mode.controlModeIndex)
        time.sleep(0.25) # Avoid aggressive switching
        


if __name__ == "__main__":
    rospy.init_node('interface')
    server = GuidanceInterface()
    rospy.spin()
