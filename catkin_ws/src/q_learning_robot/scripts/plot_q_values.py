#!/usr/bin/env python3
import pickle
import numpy as np
import cv2
import os

# === CONFIG ===
Q_TABLE_PATH = "/root/catkin_ws/src/q_learning_robot/q_table.pkl"
GRID_SIZE = 52
ACTIONS = ["forward", "turn_left", "turn_right", "stay"]
NUM_ACTIONS = len(ACTIONS)
NUM_ORIENTATIONS = 4

def load_q_table(filename):
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        exit(1)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    q_table = data.get('q_table', {})
    print(f"✅ Q-table loaded. States: {len(q_table)}")
    return q_table

def aggregate_q_values(q_table, grid_size, num_actions, num_orientations):
    """Compute mean Q-value for each (r,c) and action over orientations."""
    q_maps = np.zeros((num_actions, grid_size, grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            for a in range(num_actions):
                vals = []
                for o in range(num_orientations):
                    st = (r, c, o)
                    if st in q_table:
                        vals.append(q_table[st][a])
                if vals:
                    q_maps[a, r, c] = np.mean(vals)
                else:
                    q_maps[a, r, c] = 0.0
    return q_maps

def normalize_for_display(q_maps):
    """Normalize Q-values to 0–255 range for OpenCV visualization."""
    q_min, q_max = np.min(q_maps), np.max(q_maps)
    norm_maps = 255 * (q_maps - q_min) / (q_max - q_min + 1e-8)
    return norm_maps.astype(np.uint8)

def plot_q_maps(q_maps, actions):
    """Display one image per action using OpenCV."""
    for i, action in enumerate(actions):
        img = q_maps[i]
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img_color = cv2.resize(img_color, (520, 520), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(f"Q-values for '{action}'", img_color)
        cv2.imwrite(f"q_values_{action}.png", img_color)

    print("✅ Images saved as q_values_*.png")
    print("Press any key to close windows.")
    cv2.waitKey(0)   # Wait indefinitely
    choice = input().strip()
    while not choice == [0]:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    q_table = load_q_table(Q_TABLE_PATH)
    q_maps = aggregate_q_values(q_table, GRID_SIZE, NUM_ACTIONS, NUM_ORIENTATIONS)
    norm_maps = normalize_for_display(q_maps)
    plot_q_maps(norm_maps, ACTIONS)
