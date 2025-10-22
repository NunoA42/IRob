#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray, Marker
import tf
import math
import pickle
import os

class QLearningROS:
    def __init__(self):
        # --- Parâmetros de Configuração ---
        self.GRID_SIZE = 52
        self.GOAL_CELL = (33, 40)

        # Q-learning
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.num_episodes = 50000

        # Ações
        self.actions = ["forward", "turn_left", "turn_right", "stay"]
        self.num_actions = len(self.actions)
        self.num_orientations = 4

        # Recompensas
        self.goal_reward = 500
        self.obstacle_penalty = -100
        self.step_penalty = -1

        # Estado
        self.q_table = {}
        self.map_data = None
        self.map_info = None
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

        # Posição real
        self.robot_pos = {'x': 0.0, 'y': 0.0}
        self.robot_theta = 0.0

        # ROS
        rospy.init_node('q_learning_node')
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("Nó de Q-learning iniciado. À espera do mapa...")
        rospy.spin()

    def odom_callback(self, msg):
        self.robot_pos['x'] = msg.pose.pose.position.x
        self.robot_pos['y'] = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        self.robot_theta = yaw

    def move_robot(self, linear_vel, angular_vel, duration):
        rate = rospy.Rate(10)
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        t0 = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - t0 < duration:
            self.cmd_pub.publish(twist)
            rate.sleep()
        # Parar
        twist.linear.x = 0
        twist.angular.z = 0
        self.cmd_pub.publish(twist)

    def step(self, state, action_idx):
        action = self.actions[action_idx]
        step_dist = 0.2
        turn_angle = math.pi/4

        if action == "forward":
            self.move_robot(step_dist, 0, 0.5)
        elif action == "turn_left":
            self.move_robot(0, turn_angle, 1.0)
        elif action == "turn_right":
            self.move_robot(0, -turn_angle, 1.0)
        # stay não faz nada

        rospy.sleep(0.2)  # Dar tempo ao robô atualizar posição
        new_state = self.get_state_from_pos(self.robot_pos['x'], self.robot_pos['y'])

        done = False
        reward = self.step_penalty

        # Recompensa baseada na distância ao objetivo
        goal_x, goal_y = self.get_world_coords_from_state(self.GOAL_CELL)
        old_x, old_y = self.get_world_coords_from_state(state)
        old_dist = math.hypot(goal_x - old_x, goal_y - old_y)
        new_dist = math.hypot(goal_x - self.robot_pos['x'], goal_y - self.robot_pos['y'])

        if action == "stay":
            reward -= 5

        if (new_state[0], new_state[1]) == self.GOAL_CELL:
            reward = self.goal_reward
            done = True
        elif self.grid[new_state[0], new_state[1]] == 1:
            reward = self.obstacle_penalty
            done = True

        return new_state, reward, done

    # --- O resto do código mantém-se igual ---
    # process_map(), get_state_from_pos(), get_world_coords_from_state(), run_q_learning(), 
    # print_policy(), save_q_table(), load_q_table(), publish_visualization(), publish_tf(), 
    # extract_policy_grid(), save_progress_to_csv(), track_q_values() permanecem sem alteração

if __name__ == '__main__':
    try:
        QLearningROS()
    except rospy.ROSInterruptException:
        # Parar robô em caso de interrupção
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        pub.publish(twist)
        pass
