#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

# File paths
BASE_DIR = "/root/catkin_ws/src/q_learning_robot"
SUMMARY_FILE = os.path.join(BASE_DIR, "episodes_summary.csv")
VISITS_FILE = os.path.join(BASE_DIR, "visit_counts.npy")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "q_snapshots")

def plot_curves():
    """Read episodes_summary.csv and plot reward and success rate curves."""
    episodes, rewards, success, success_rate = [], [], [], []
    if not os.path.exists(SUMMARY_FILE):
        print(f" File {SUMMARY_FILE} not found.")
        return
    
    with open(SUMMARY_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["Episode"]))
            rewards.append(float(row["Total_Reward"]))
            success.append(int(row["Success"]))
            success_rate.append(float(row["Success_Rate_100ep"]))

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episodes, rewards, label="Total Reward", linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Curve per Episode")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, success_rate, label="Success Rate (100-ep moving avg)", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate Curve")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_visit_heatmap():
    """Show visit counts heatmap."""
    if not os.path.exists(VISITS_FILE):
        print(f" File {VISITS_FILE} not found.")
        return
    
    visits = np.load(VISITS_FILE)
    visits_log = np.log1p(visits)
    visits_norm = (visits_log / np.max(visits_log) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(visits_norm, cv2.COLORMAP_JET)

    cv2.namedWindow("Visit Heatmap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visit Heatmap", 800, 600)

    cv2.imshow("Visit Heatmap", heatmap)
    print("Press any key to close the heatmap...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_q_snapshots():
    """Show Q-learning evolution statistics from snapshots."""
    if not os.path.exists(SNAPSHOT_DIR):
        print("Snapshots directory not found.")
        return
    
    snapshot_files = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".pkl")])
    if not snapshot_files:
        print(" No snapshots found.")
        return

    avg_q_values = []
    max_q_values = []
    episodes = []

    for f in snapshot_files:
        path = os.path.join(SNAPSHOT_DIR, f)
        with open(path, 'rb') as file:
            q_table = pickle.load(file)
        all_q = [np.mean(v) for v in q_table.values()]
        avg_q_values.append(np.mean(all_q))
        max_q_values.append(np.max([np.max(v) for v in q_table.values()]))
        ep_num = int(''.join(filter(str.isdigit, f)))
        episodes.append(ep_num)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_q_values, label="Average Q", linewidth=1.5)
    plt.plot(episodes, max_q_values, label="Max Q", linewidth=1.5)
    plt.xlabel("Episode (snapshot)")
    plt.ylabel("Q-value")
    plt.title("Q-values Evolution During Training")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("=== 🔍 Q-Learning ROS Analyzer ===")
    print("Reward and success curves")
    print("Visit heatmap")
    print("Q-value evolution (snapshots)")
    print("Choose option (1/2/3 or ENTER for all): ", end="")
    choice = input().strip()

    if choice in ["1", ""]:
        plot_curves()
    if choice in ["2", ""]:
        plot_visit_heatmap()
    if choice in ["3", ""]:
        analyze_q_snapshots()