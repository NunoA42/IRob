#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path

path = Path()

def callback(msg):
    global path
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose
    pose.pose.position.z = 0.0
    path.header = msg.header
    path.poses.append(pose)
    pub.publish(path)

if __name__ == "__main__":
    rospy.init_node("ground_truth_path")
    pub = rospy.Publisher("/ground_truth_path", Path, queue_size=10)
    rospy.Subscriber("/ground_truth_downsampled", PoseWithCovarianceStamped, callback)
    rospy.spin()
