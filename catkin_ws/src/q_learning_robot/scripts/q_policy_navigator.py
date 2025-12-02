#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pickle
import numpy as np

import rospy
import actionlib

from std_msgs.msg import Int32MultiArray
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf


class QPolicyNavigator:
    def __init__(self):
        # ROS parameters
        self.q_table_path = rospy.get_param('~q_table_path', '/home/acsdc/Desktop/IRob-main/catkin_ws/src/q_learning_robot/q_table.pkl')
        self.grid_size = int(rospy.get_param('~grid_size', 52))
        self.num_orientations = int(rospy.get_param('~num_orientations', 4))
        self.goal_timeout = float(rospy.get_param('~goal_timeout', 60.0))
        self.retry_limit = int(rospy.get_param('~retry_limit', 2))
        self.frame_map = rospy.get_param('~frame_map', 'map')
        self.pos_tolerance = float(rospy.get_param('~pos_tolerance', 0.15))
        self.yaw_tolerance_deg = float(rospy.get_param('~yaw_tolerance_deg', 15.0))
        self.yaw_tolerance = math.radians(self.yaw_tolerance_deg)

        # Map and pose state
        self.map_info = None
        self.have_map = False
        self.pose_xytheta = None  # (x, y, yaw)

        # Q-table
        self.q_table = {}
        self.actions = ['forward', 'turn_left', 'turn_right', 'stay']
        self.load_q_table(self.q_table_path)

        # ROS I/O
        rospy.Subscriber('/map', OccupancyGrid, self.cb_map, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.cb_pose, queue_size=10)
        rospy.Subscriber('/q_target_cell', Int32MultiArray, self.cb_external_target, queue_size=10)
        self.decided_pub = rospy.Publisher('/q_decided_cell', Int32MultiArray, queue_size=10)

        # move_base action client
        self.mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("[QPolicyNavigator] Waiting for move_base server...")
        self.mb_client.wait_for_server()
        rospy.loginfo("[QPolicyNavigator] Connected to move_base.")

        # Main timer loop
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_step)

        # Cache last target to avoid spam
        self.last_target_cell = None

    def cb_map(self, msg: OccupancyGrid):
        if not self.have_map:
            self.map_info = msg.info
            self.have_map = True
            rospy.loginfo("[QPolicyNavigator] Map received: %dx%d, res=%.3fm",
                          msg.info.width, msg.info.height, msg.info.resolution)

    def cb_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose_xytheta = (p.x, p.y, self.norm_angle(yaw))

    def cb_external_target(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            rospy.logwarn("[QPolicyNavigator] Invalid /q_target_cell (needs [row, col]).")
            return
        r, c = int(msg.data[0]), int(msg.data[1])
        rospy.loginfo("[QPolicyNavigator] External target received: (%d,%d)", r, c)
        self.navigate_to_cell((r, c))

    def timer_step(self, _event):
        if not (self.have_map and self.pose_xytheta and self.q_table):
            return

        # Get current state
        cur_cell = self.world_to_cell(self.pose_xytheta[0], self.pose_xytheta[1])
        orient_idx = self.get_orientation_index(self.pose_xytheta[2])
        state = (cur_cell[0], cur_cell[1], orient_idx)

        if state not in self.q_table:
            return

        # Choose greedy action
        qvals = self.q_table[state]
        action_idx = int(np.argmax(qvals))
        action = self.actions[action_idx]

        # Determine next cell based on action
        desired_yaw = self.pose_xytheta[2]
        if action == 'turn_left':
            desired_yaw = self.norm_angle(desired_yaw + math.pi/2)
        elif action == 'turn_right':
            desired_yaw = self.norm_angle(desired_yaw - math.pi/2)

        next_cell = cur_cell if action == 'stay' else self.pick_best_neighbor_towards_yaw(cur_cell, desired_yaw)

        if next_cell is None:
            return

        # Publish for debug
        out = Int32MultiArray(data=[int(next_cell[0]), int(next_cell[1])])
        self.decided_pub.publish(out)

        # Avoid resending same goal
        if self.last_target_cell != next_cell:
            rospy.loginfo("[QPolicyNavigator] Action=%s -> next cell %s", action, next_cell)
            self.navigate_to_cell(next_cell)
            self.last_target_cell = next_cell

    def navigate_to_cell(self, cell_rc):
        if not self.have_map or self.pose_xytheta is None:
            rospy.logwarn("[QPolicyNavigator] Waiting for /map and /amcl_pose to navigate...")
            return

        r, c = int(cell_rc[0]), int(cell_rc[1])
        goal_xy = self.cell_to_world_center(r, c)

        # Calculate yaw pointing to target
        dx = goal_xy[0] - self.pose_xytheta[0]
        dy = goal_xy[1] - self.pose_xytheta[1]
        yaw = math.atan2(dy, dx)

        # If already within tolerance, consider reached
        if self.within_tolerance(goal_xy[0], goal_xy[1], yaw):
            rospy.loginfo("[QPolicyNavigator] Already within tolerance of cell (%d,%d).", r, c)
            return

        # Send goal to move_base with continuous tolerance checking
        for attempt in range(self.retry_limit + 1):
            ok = self.send_move_base_goal(goal_xy[0], goal_xy[1], yaw)
            if ok:
                rospy.loginfo("[QPolicyNavigator] Reached cell (%d,%d).", r, c)
                return
            else:
                rospy.logwarn("[QPolicyNavigator] Failed cell (%d,%d), attempt %d/%d.",
                              r, c, attempt + 1, self.retry_limit + 1)
        rospy.logerr("[QPolicyNavigator] Gave up on cell (%d,%d).", r, c)

    def send_move_base_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.frame_map
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        # If already within tolerance, return success
        if self.within_tolerance(x, y, yaw):
            return True

        self.mb_client.send_goal(goal)

        # Monitor pose and apply tolerances while goal is active
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < self.goal_timeout:
            if self.within_tolerance(x, y, yaw):
                try:
                    self.mb_client.cancel_goal()
                except Exception:
                    pass
                return True
            state = self.mb_client.get_state()
            if state == 3:
                return True
            rate.sleep()

        # Timeout - cancel and fail
        rospy.logwarn("[QPolicyNavigator] Goal timeout (%.1fs). Cancelling...", self.goal_timeout)
        self.mb_client.cancel_goal()
        return False

    def within_tolerance(self, gx, gy, gyaw):
        if self.pose_xytheta is None:
            return False
        x, y, yaw = self.pose_xytheta
        dist = math.hypot(gx - x, gy - y)
        dyaw = self.norm_angle(gyaw - yaw)
        return (dist <= self.pos_tolerance) and (abs(dyaw) <= self.yaw_tolerance)

    def cell_to_world_center(self, row, col):
        """Convert cell (r,c) to world coordinates."""
        cell_w = self.map_info.width / float(self.grid_size)
        cell_h = self.map_info.height / float(self.grid_size)
        x = (col + 0.5) * cell_w * self.map_info.resolution + self.map_info.origin.position.x
        y = (row + 0.5) * cell_h * self.map_info.resolution + self.map_info.origin.position.y
        return (x, y)

    def world_to_cell(self, x, y):
        col = int((x - self.map_info.origin.position.x) / self.map_info.resolution / (self.map_info.width / float(self.grid_size)))
        row = int((y - self.map_info.origin.position.y) / self.map_info.resolution / (self.map_info.height / float(self.grid_size)))
        row = max(0, min(self.grid_size - 1, row))
        col = max(0, min(self.grid_size - 1, col))
        return (row, col)

    def pick_best_neighbor_towards_yaw(self, cur_cell, desired_yaw):
        """Choose 4-connected neighbor that best aligns with desired_yaw."""
        r, c = cur_cell
        candidates = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        hx, hy = math.cos(desired_yaw), math.sin(desired_yaw)
        best = None
        best_dot = -1e9
        cx, cy = self.cell_to_world_center(r, c)
        for rr, cc in candidates:
            if rr < 0 or rr >= self.grid_size or cc < 0 or cc >= self.grid_size:
                continue
            nx, ny = self.cell_to_world_center(rr, cc)
            vx, vy = (nx - cx), (ny - cy)
            norm = math.hypot(vx, vy)
            if norm < 1e-6:
                continue
            dot = (vx / norm) * hx + (vy / norm) * hy
            if dot > best_dot:
                best_dot = dot
                best = (rr, cc)
        return best

    def get_orientation_index(self, theta):
        """Discretize yaw into orientation buckets."""
        frac = (theta + math.pi) / (2 * math.pi)
        idx = int(round(frac * self.num_orientations)) % self.num_orientations
        return idx

    @staticmethod
    def norm_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def load_q_table(self, path):
        if not os.path.exists(path):
            rospy.logwarn("[QPolicyNavigator] Q-table not found at %s - external target mode only.", path)
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data.get('q_table', {})
        rospy.loginfo("[QPolicyNavigator] Q-table loaded (%d states).", len(self.q_table))


def main():
    rospy.init_node('q_policy_navigator')
    node = QPolicyNavigator()
    rospy.loginfo("[QPolicyNavigator] Ready. Publish /q_target_cell to force next square or let it follow Q-policy.")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass