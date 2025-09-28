#!/usr/bin/env python
import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped

def main():
    rospy.init_node("tf_to_pose_downsampled")
    pub = rospy.Publisher("/ground_truth_downsampled", PoseWithCovarianceStamped, queue_size=10)

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(1.0)  # publish at 1 Hz
    while not rospy.is_shutdown():
        try:
            # Lookup the transform mocap -> mocap_laser_link
            trans = tf_buffer.lookup_transform("mocap", "mocap_laser_link", rospy.Time(0))
            
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "mocap"
            msg.pose.pose.position.x = trans.transform.translation.x
            msg.pose.pose.position.y = trans.transform.translation.y
            msg.pose.pose.position.z = trans.transform.translation.z
            msg.pose.pose.orientation = trans.transform.rotation

            # Give it a small covariance (so EKF can use it)
            msg.pose.covariance = [0.0]*36
            msg.pose.covariance[0] = 1e-4   # x
            msg.pose.covariance[7] = 1e-4   # y
            msg.pose.covariance[35] = 1e-4  # yaw

            pub.publish(msg)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            pass
        rate.sleep()

if __name__ == "__main__":
    main()
