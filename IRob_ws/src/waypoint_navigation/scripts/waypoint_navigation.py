#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from std_srvs.srv import Empty

# ==============================
# User configuration
# ==============================

# Waypoints (edit to match your map)
WAYPOINTS = [
    Pose(Point(4.560, -2.100, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
    Pose(Point(0.44, -1.87, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
    Pose(Point(1.736,  2.362, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0)),
    Pose(Point(1.836,  0.359, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
]

# Maximum time to allow per attempt (seconds)
GOAL_TIMEOUT = 180   # 3 minutes per try
# Number of maximum attempts per waypoint
MAX_TRIES = 4


# ==============================
# Helper functions
# ==============================

def clear_costmaps():
    """Call the /move_base/clear_costmaps service to reset costmaps."""
    try:
        rospy.wait_for_service('/move_base/clear_costmaps', timeout=5)
        clear = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        clear()
        rospy.loginfo("Costmaps cleared.")
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logwarn(f"Failed to clear costmaps: {e}")


def send_goal(client, pose, frame="map"):
    """Send a goal to move_base and wait for result."""
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = frame
    # Backdate timestamp slightly to avoid TF extrapolation errors
    goal.target_pose.header.stamp = rospy.Time.now() - rospy.Duration(0.3)
    goal.target_pose.pose = pose

    client.send_goal(goal)
    finished = client.wait_for_result(rospy.Duration(GOAL_TIMEOUT))

    if not finished:
        client.cancel_goal()
        rospy.logwarn("Timed out achieving goal.")
        return False

    state = client.get_state()
    if state == 3:
        rospy.loginfo("Goal succeeded!")
        return True
    else:
        rospy.logwarn(f"Goal failed with state: {state}")
        return False


def request_nomotion_update():
    """Force AMCL to update map->odom transform with current data."""
    try:
        rospy.wait_for_service('/request_nomotion_update', timeout=2.0)
        nomotion = rospy.ServiceProxy('/request_nomotion_update', Empty)
        nomotion()
        rospy.loginfo("AMCL no-motion update triggered.")
    except (rospy.ServiceException, rospy.ROSException):
        rospy.logwarn("No-motion AMCL update failed or timed out.")


def global_localization(spin=True):
    """Spread AMCL particles randomly for global re-localization."""
    try:
        rospy.wait_for_service('/global_localization', timeout=5)
        gloc = rospy.ServiceProxy('/global_localization', Empty)
        gloc()
        rospy.loginfo("Triggered AMCL global localization (particle spread).")

        if spin:
            spin_robot()

    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logwarn(f"Global localization failed: {e}")


def spin_robot(duration=10.0, angular_speed=0.3):
    """Rotate the robot in place to help AMCL resample."""
    rospy.loginfo("Spinning robot to help AMCL converge...")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
    twist = Twist()
    twist.angular.z = angular_speed

    rate = rospy.Rate(10)
    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time).to_sec() < duration and not rospy.is_shutdown():
        pub.publish(twist)
        rate.sleep()

    # Stop rotation
    twist.angular.z = 0.0
    pub.publish(twist)
    rospy.loginfo("Spin complete.")


# ==============================
# Main execution loop
# ==============================

if __name__ == "__main__":
    rospy.init_node("waypoint_navigation")

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base.")

    for i, wp in enumerate(WAYPOINTS, start=1):
        rospy.loginfo(f"Preparing waypoint {i}: {wp.position.x}, {wp.position.y}")

        # Give AMCL and TF a short time to stabilize after previous goal
        rospy.sleep(2.0)

        # Clear old costmap data to prevent false obstacle detections
        clear_costmaps()
        rospy.sleep(0.1)

        # Force AMCL to update its transform before planning
        request_nomotion_update()
        rospy.sleep(0.3)

        # --- Retry + recovery logic ---
        success = False
        for attempt in range(MAX_TRIES):
            rospy.loginfo(f"Attempt {attempt+1}/{MAX_TRIES} for waypoint {i}")
            clear_costmaps()

            success = send_goal(client, wp)
            if success:
                rospy.loginfo(f"Waypoint {i} reached successfully on attempt {attempt+1}.")
                break

            rospy.logwarn(f"Attempt {attempt+1}/{MAX_TRIES} failed for waypoint {i}")

            # Different actions per attempt
            if attempt == 0:
                rospy.sleep(2.0)
            elif attempt == 1:
                rospy.logwarn("Triggering AMCL global localization (particle spread)...")
                global_localization(spin=True)
                rospy.sleep(3.0)
                request_nomotion_update()
                rospy.sleep(0.3)
            elif attempt == 2:
                rospy.sleep(2.0)

        if not success:
            rospy.logwarn(f"Failed to reach waypoint {i} after {MAX_TRIES} attempts. Stopping sequence.")
            break

    rospy.loginfo("Waypoint navigation sequence complete.")
