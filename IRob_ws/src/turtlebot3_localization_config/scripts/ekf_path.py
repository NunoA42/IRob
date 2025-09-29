#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class EkfPath:
    def __init__(self):
        self.pub = rospy.Publisher("/ekf_path", Path, queue_size=10)
        self.path = Path()
        self.path.header.frame_id = "map"

        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        pose.pose.position.z = 0.0

        # Update path
        self.path.header.stamp = msg.header.stamp
        self.path.poses.append(pose)

        self.pub.publish(self.path)

if __name__ == "__main__":
    rospy.init_node("ekf_path_node")
    EkfPath()
    rospy.spin()
