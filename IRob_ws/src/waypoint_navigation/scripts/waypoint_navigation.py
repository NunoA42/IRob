#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion

# List of waypoints (edit to match your map coordinates)
WAYPOINTS = [
    Pose(Point(5.060, -2.441, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
    Pose(Point(1.926, -2.626, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
    Pose(Point(1.736,  2.362, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
    Pose(Point(1.836,  0.359, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
]

def send_goal(client, pose, frame="map"):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = frame
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = pose

    client.send_goal(goal)
    finished = client.wait_for_result(rospy.Duration(60))  # timeout (seconds)

    if not finished:
        client.cancel_goal()
        rospy.logwarn("Timed out achieving goal")
        return False

    state = client.get_state()
    if state == 3:
        rospy.loginfo("Goal succeeded!")
        return True
    else:
        rospy.logwarn(f"Goal failed with state: {state}")
        return False

if __name__ == "__main__":
    rospy.init_node("waypoint_navigation")

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base.")

    for i, wp in enumerate(WAYPOINTS, start=1):
        rospy.loginfo(f"Sending waypoint {i}: {wp.position.x}, {wp.position.y}")
        success = send_goal(client, wp)
        if not success:
            rospy.logwarn(f"Failed to reach waypoint {i}.")
            break

    rospy.loginfo("Navigation finished.")
