#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
import tf
import math
import pickle
import os

class QLearningROS:
    def __init__(self):
        # Configuration parameters
        self.GRID_SIZE = 52
        self.GOAL_CELL = (33, 36)

        # Q-Learning parameters
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.99996
        self.epsilon_min = 0.05
        self.num_episodes = 1000

        # Actions: 0: forward, 1: turn left, 2: turn right, 3: stay
        self.actions = ["forward", "turn_left", "turn_right", "stay"]
        self.num_actions = len(self.actions)
        self.num_orientations = 4  # 4 directions (N,E,S,W)

        # Rewards
        self.goal_reward = 1000
        self.obstacle_penalty = -100
        self.step_penalty = -5

        # State variables
        self.q_table = {}
        self.map_data = None
        self.map_info = None
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))  # 0: free, 1: obstacle

        # Simulated robot variables
        self.robot_pos = {'x': 0.0, 'y': 0.0}
        self.robot_theta = 0.0

        # Episode statistics
        self.episode_rewards = []
        self.episode_success = []

        # Visit counter
        self.visit_counts = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)

        # Snapshots directory
        self.snapshot_dir = "/root/catkin_ws/src/q_learning_robot/q_snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # ROS initialization
        rospy.init_node('q_learning_node')

        # Subscribers and Publishers
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("Q-learning node started. Waiting for map...")
        rospy.spin()

    def map_callback(self, msg):
        if self.map_data is not None:
            return  # Process map only once

        rospy.loginfo("Map received!")
        self.map_data = msg.data
        self.map_info = msg.info
        
        self.process_map()
        self.run_q_learning()

    def process_map(self):
        """Convert ROS OccupancyGrid to internal grid."""
        map_width = self.map_info.width
        map_height = self.map_info.height

        cell_width_map = map_width / self.GRID_SIZE
        cell_height_map = map_height / self.GRID_SIZE

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                is_obstacle = False
                for i in range(int(r * cell_height_map), int((r + 1) * cell_height_map)):
                    for j in range(int(c * cell_width_map), int((c + 1) * cell_width_map)):
                        map_index = i * map_width + j
                        if self.map_data[map_index] > 99:
                            is_obstacle = True
                            break
                    if is_obstacle:
                        break
                
                if is_obstacle:
                    self.grid[r, c] = 1

        # Ensure goal cell is not an obstacle
        if self.grid[self.GOAL_CELL] == 1:
            self.grid[self.GOAL_CELL] = 0
            rospy.logwarn("Goal cell was marked as obstacle. Cleared.")
        
        # World boundaries
        limite_x_min = -0.9
        limite_x_max = 5.5
        limite_y_min = -3.1
        limite_y_max = 2.9

        cell_width_m = (self.map_info.width / self.GRID_SIZE) * self.map_info.resolution
        cell_height_m = (self.map_info.height / self.GRID_SIZE) * self.map_info.resolution

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                x_celula = c * cell_width_m + self.map_info.origin.position.x
                y_celula = r * cell_height_m + self.map_info.origin.position.y

                if (
                    x_celula <= limite_x_min or
                    x_celula >= limite_x_max or
                    y_celula <= limite_y_min or
                    y_celula >= limite_y_max
                ):
                    self.grid[r, c] = 1

        # Define obstacle rectangles
        x1_min, x1_max = -1.5, 0.7
        y1_min, y1_max = 1.85, 4.3

        x2_min, x2_max = 0.14, 3.9
        y2_min, y2_max = -1.3, -0.14

        x3_min, x3_max = -0.65, 3.84
        y3_min, y3_max = 0.855, 1.66

        x4_min, x4_max = 4.876, 6.364
        y4_min, y4_max = -3.263, 3.281

        x5_min, x5_max = -0.766, 3.8
        y5_min, y5_max = 0.633, 1.438

        cell_width_m = (self.map_info.width / self.GRID_SIZE) * self.map_info.resolution
        cell_height_m = (self.map_info.height / self.GRID_SIZE) * self.map_info.resolution

        # Mark obstacles within rectangles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                x_celula = c * cell_width_m + self.map_info.origin.position.x
                y_celula = r * cell_height_m + self.map_info.origin.position.y
                
                if (
                    (x1_min <= x_celula <= x1_max and y1_min <= y_celula <= y1_max) or
                    (x2_min <= x_celula <= x2_max and y2_min <= y_celula <= y2_max) or
                    (x3_min <= x_celula <= x3_max and y3_min <= y_celula <= y3_max) or
                    (x4_min <= x_celula <= x4_max and y4_min <= y_celula <= y4_max) or
                    (x5_min <= x_celula <= x5_max and y5_min <= y_celula <= y5_max)
                ):
                    self.grid[r, c] = 1
                    
        # Force specific cells to be free
        coordenadas_livres = [
            (1.621452808380127, -2.0439205169677734),
            (2.0016815662384033, -2.0534446239471436)
        ]

        for (x, y) in coordenadas_livres:
            c = int((x - self.map_info.origin.position.x) / self.map_info.resolution / (self.map_info.width / self.GRID_SIZE))
            r = int((y - self.map_info.origin.position.y) / self.map_info.resolution / (self.map_info.height / self.GRID_SIZE))
            
            r = max(0, min(self.GRID_SIZE - 1, r))
            c = max(0, min(self.GRID_SIZE - 1, c))
            
            self.grid[r, c] = 0

        rospy.loginfo("Map processed to %dx%d grid.", self.GRID_SIZE, self.GRID_SIZE)

    def get_orientation_index(self):
        """Discretize robot_theta into orientation buckets."""
        theta = self.robot_theta
        frac = (theta + math.pi) / (2 * math.pi)
        idx = int(round(frac * self.num_orientations)) % self.num_orientations
        return idx

    def get_state_from_pos(self, x, y):
        """Return state (r, c, orient_idx) from world coordinates."""
        grid_c = int((x - self.map_info.origin.position.x) / self.map_info.resolution / (self.map_info.width / self.GRID_SIZE))
        grid_r = int((y - self.map_info.origin.position.y) / self.map_info.resolution / (self.map_info.height / self.GRID_SIZE))
        grid_r = max(0, min(self.GRID_SIZE - 1, grid_r))
        grid_c = max(0, min(self.GRID_SIZE - 1, grid_c))
        orient = self.get_orientation_index()
        return (grid_r, grid_c, orient)     

    def get_world_coords_from_state(self, state):
        """Convert grid state to world coordinates (x, y)."""
        if len(state) == 3:
            row, col = state[0], state[1]
        else:
            row, col = state
        cell_width_map = self.map_info.width / self.GRID_SIZE
        cell_height_map = self.map_info.height / self.GRID_SIZE
        
        x = (col + 0.5) * cell_width_map * self.map_info.resolution + self.map_info.origin.position.x
        y = (row + 0.5) * cell_height_map * self.map_info.resolution + self.map_info.origin.position.y
        return x, y

    def get_random_free_state(self):
        """Find a random free cell to start an episode."""
        while True:
            r,c = (23, 39)
            r, c = np.random.randint(0, self.GRID_SIZE, size=2)
            if self.grid[r, c] == 0:
                return (r, c)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.q_table.get(state, np.zeros(self.num_actions))
            return np.argmax(q_values)

    def step(self, state, action_idx):
        """Simulate robot step and return new state, reward, and done flag."""
        action = self.actions[action_idx]
        
        step_dist = 0.2
        turn_angle = math.pi / 2

        if action == "forward":
            self.robot_pos['x'] += step_dist * math.cos(self.robot_theta)
            self.robot_pos['y'] += step_dist * math.sin(self.robot_theta)
        elif action == "turn_left":
            self.robot_theta += turn_angle
            self.robot_pos['x'] += step_dist * math.cos(self.robot_theta)
            self.robot_pos['y'] += step_dist * math.sin(self.robot_theta)
        elif action == "turn_right":
            self.robot_theta -= turn_angle
            self.robot_pos['x'] += step_dist * math.cos(self.robot_theta)
            self.robot_pos['y'] += step_dist * math.sin(self.robot_theta)

        # Normalize angle
        self.robot_theta = math.atan2(math.sin(self.robot_theta), math.cos(self.robot_theta))
        
        new_state = self.get_state_from_pos(self.robot_pos['x'], self.robot_pos['y'])
        
        done = False
        reward = self.step_penalty

        goal_x, goal_y = self.get_world_coords_from_state(self.GOAL_CELL)
        old_x, old_y = self.get_world_coords_from_state(state)
        old_dist = math.hypot(goal_x - old_x, goal_y - old_y)
        new_dist = math.hypot(goal_x - self.robot_pos['x'], goal_y - self.robot_pos['y'])

        phi_old = -old_dist

        if action == "stay":
            reward -= 50

        # Check episode termination
        if (new_state[0], new_state[1]) == self.GOAL_CELL:
            reward = self.goal_reward
            done = True
        elif self.grid[new_state[0], new_state[1]] == 1:
            reward = self.obstacle_penalty
            done = True

        return new_state, reward, done

    def run_q_learning(self):
        """Main Q-learning training loop."""
        rospy.loginfo("Starting Q-learning training...")
        
        self.q_value_progress = []
        self.policy_snapshots = []

        self.load_q_table()

        for episode in range(self.num_episodes):
            initial_state = self.get_random_free_state()
            self.robot_pos['x'], self.robot_pos['y'] = self.get_world_coords_from_state(initial_state)

            state = self.get_state_from_pos(self.robot_pos['x'], self.robot_pos['y'])

            self.track_q_values(episode)

            done = False
            total_reward = 0
            max_steps = 1000
            success = False
            
            for step_num in range(max_steps):
                action = self.choose_action(state)
                
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(self.num_actions)
                
                next_state, reward, done = self.step(state, action)
                total_reward += reward

                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(self.num_actions)

                old_value = self.q_table[state][action]
                next_max = np.max(self.q_table[next_state])
                
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state][action] = new_value
                
                state = next_state

                r, c = state[0], state[1]
                self.visit_counts[r, c] += 1

                self.publish_visualization(state)
                self.publish_tf()

                if done:
                    if (next_state[0], next_state[1]) == self.GOAL_CELL:
                        success = True
                    break

            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(total_reward)
            self.episode_success.append(success)

            # Save Q-table snapshot
            if (episode + 1) % 5000 == 0:
                snapshot_path = os.path.join(self.snapshot_dir, f"q_snapshot_ep{episode+1}.pkl")
                with open(snapshot_path, "wb") as f:
                    pickle.dump(self.q_table, f)

            # Periodic logging and saving
            if (episode + 1) % 10000 == 0:
                rospy.loginfo("Episode %d/%d completed. Total reward: %.2f. Epsilon: %.3f", 
                              episode + 1, self.num_episodes, total_reward, self.epsilon)
                self.save_q_table()
                self.policy_snapshots.append((episode + 1, self.extract_policy_grid()))

        rospy.loginfo("Training completed.")
        self.save_q_table()
        rospy.loginfo("Final Q-table saved.")
        self.print_policy()
        self.publish_visualization(self.GOAL_CELL)
        self.save_progress_to_csv()
        self.save_episode_summary()
        self.save_visit_map()

    def print_policy(self):
        """Print learned policy to terminal."""
        rospy.loginfo("--- Learned Optimal Policy ---")
        policy_grid = [[' ' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 1:
                    policy_grid[r][c] = '#'
                elif (r, c) == self.GOAL_CELL:
                    policy_grid[r][c] = 'G'
                else:
                    aggregated = np.zeros(self.num_actions)
                    found_any = False
                    for o in range(self.num_orientations):
                        st = (r, c, o)
                        if st in self.q_table:
                            aggregated = np.maximum(aggregated, self.q_table[st])
                            found_any = True
                    if found_any:
                        best_action_idx = np.argmax(aggregated)
                        action_symbols = ['^', '<', '>', 'o']
                        policy_grid[r][c] = action_symbols[best_action_idx]
        
        for row in policy_grid:
            print(" ".join(row))
        rospy.loginfo("-----------------------------")
    
    def save_q_table(self, filename="/root/catkin_ws/src/q_learning_robot/q_table.pkl"):
        """Save Q-table and training parameters to file."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        rospy.loginfo("Q-table saved to %s", filename)

    def load_q_table(self, filename="/root/catkin_ws/src/q_learning_robot/q_table.pkl"):
        """Load Q-table and training parameters from file if exists."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon)
                self.alpha = data.get('alpha', self.alpha)
                self.gamma = data.get('gamma', self.gamma)
            rospy.loginfo("Q-table loaded from %s", filename)
        else:
            rospy.logwarn("No Q-table found at %s. Starting from scratch.", filename)

    def publish_visualization(self, robot_current_state):
        """Publish colored grid as MarkerArray to RViz."""
        marker_array = MarkerArray()
        marker_id = 0
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "grid_cells"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                x, y = self.get_world_coords_from_state((r, c))
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = -0.01

                cell_width_world = (self.map_info.width / self.GRID_SIZE) * self.map_info.resolution
                cell_height_world = (self.map_info.height / self.GRID_SIZE) * self.map_info.resolution
                marker.scale.x = cell_width_world
                marker.scale.y = cell_height_world
                marker.scale.z = 0.01

                marker.color.a = 0.7

                if self.grid[r, c] == 1:
                    marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0)
                elif (r, c) == self.GOAL_CELL:
                    marker.color.r, marker.color.g, marker.color.b = (0.0, 1.0, 0.0)
                else:
                    if isinstance(robot_current_state, tuple) and len(robot_current_state) >= 2 and (r, c) == (robot_current_state[0], robot_current_state[1]):
                        marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.0)
                    else:
                        marker.color.r, marker.color.g, marker.color.b = (0.0, 0.0, 1.0)

                marker_array.markers.append(marker)
                marker_id += 1
        
        self.marker_pub.publish(marker_array)

    def publish_tf(self):
        """Publish simulated robot pose transform."""
        self.tf_broadcaster.sendTransform(
            (self.robot_pos['x'], self.robot_pos['y'], 0),
            tf.transformations.quaternion_from_euler(0, 0, self.robot_theta),
            rospy.Time.now(),
            "base_link",
            "odom"
        )

    def extract_policy_grid(self):
        """Return learned policy grid for saving or analysis."""
        policy_grid = [[' ' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 1:
                    policy_grid[r][c] = '#'
                elif (r, c) == self.GOAL_CELL:
                    policy_grid[r][c] = 'G'
                else:
                    aggregated = np.zeros(self.num_actions)
                    found_any = False
                    for o in range(self.num_orientations):
                        st = (r, c, o)
                        if st in self.q_table:
                            aggregated = np.maximum(aggregated, self.q_table[st])
                            found_any = True
                    if found_any:
                        best_action_idx = np.argmax(aggregated)
                        symbols = ['^', '<', '>', 'o']
                        policy_grid[r][c] = symbols[best_action_idx]
        return policy_grid

    def save_progress_to_csv(self, q_values_file="/root/catkin_ws/src/q_learning_robot/q_values_progress.csv",
                             policies_file="/root/catkin_ws/src/q_learning_robot/policies_progress.csv"):
        """Save Q-values evolution and policy snapshots to CSV files."""
        import csv

        with open(q_values_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Average_Q_Value", "Max_Q_Value"])
            for ep, avg_q, max_q in self.q_value_progress:
                writer.writerow([ep, avg_q, max_q])

        with open(policies_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Policy_Grid"])
            for ep, grid in self.policy_snapshots:
                grid_flat = ["".join(row) for row in grid]
                writer.writerow([ep, "|".join(grid_flat)])

        rospy.loginfo("Q-values and policies progress saved to CSV.")

    def track_q_values(self, episode):
        """Compute average and max Q-values to monitor convergence."""
        if not self.q_table:
            avg_q_value = 0.0
            max_q_value = 0.0
        else:
            q_values_all = [np.mean(v) for v in self.q_table.values()]
            avg_q_value = np.mean(q_values_all)
            max_q_value = np.max([np.max(v) for v in self.q_table.values()])

        self.q_value_progress.append((episode, avg_q_value, max_q_value))

    def save_episode_summary(self, filename="/root/catkin_ws/src/q_learning_robot/episodes_summary.csv"):
        """Save average reward and success rate per episode."""
        import csv
        success_rate = []
        window = 100
        for i in range(len(self.episode_rewards)):
            recent_success = self.episode_success[max(0, i-window):i+1]
            success_rate.append(sum(recent_success) / len(recent_success))
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Total_Reward", "Success", "Success_Rate_100ep"])
            for ep, rew, suc, rate in zip(range(len(self.episode_rewards)), 
                                        self.episode_rewards, 
                                        self.episode_success,
                                        success_rate):
                writer.writerow([ep+1, rew, int(suc), rate])
        
        rospy.loginfo("Episode summary saved to %s", filename)

    def save_visit_map(self, filename="/root/catkin_ws/src/q_learning_robot/visit_counts.npy"):
        """Save visit counts map."""
        np.save(filename, self.visit_counts)
        rospy.loginfo("Visit map saved to %s", filename)


if __name__ == '__main__':
    try:
        QLearningROS()
    except rospy.ROSInterruptException:
        pass