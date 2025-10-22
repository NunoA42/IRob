#!/usr/bin/env python3
import pickle
import os
import numpy as np
import yaml
from PIL import Image

# === CONFIGURATION ===
QTABLE_PATH = "/root/catkin_ws/src/q_learning_robot/q_table.pkl"
MAP_PATH = "/root/catkin_ws/src/q_learning_robot/maps/my_map1.yaml"
GRID_SIZE = 52              # Adjust if your Q-table grid size differs
GOAL_CELL = (33, 33)        # Adjust according to your environment


# === LOAD Q-TABLE ===
def load_q_table(path):
    if not os.path.exists(path):
        print(f"[ERRO] Ficheiro não encontrado: {path}")
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('q_table', {})


# === LOAD MAP FROM YAML ===
def load_map_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"[ERRO] Mapa não encontrado: {yaml_path}")
        return None

    with open(yaml_path, 'r') as f:
        map_yaml = yaml.safe_load(f)

    image_path = os.path.join(os.path.dirname(yaml_path), map_yaml['image'])
    if not os.path.exists(image_path):
        print(f"[ERRO] Imagem não encontrada: {image_path}")
        return None

    # Load image as grayscale
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)

    # Extract map parameters
    occupied_thresh = map_yaml.get('occupied_thresh', 0.65)
    free_thresh = map_yaml.get('free_thresh', 0.196)
    negate = map_yaml.get('negate', 0)

    if negate:
        img_data = 255 - img_data

    # Normalize to [0, 1]
    img_norm = img_data / 255.0

    # Convert to binary grid (1 = obstacle, 0 = free)
    grid = np.zeros_like(img_norm, dtype=int)

    # Obstáculos são apenas os pixels bem escuros (p.e. < 0.4)
    grid[img_norm < 0.5] = 1  # Obstáculo

    # Tudo o resto é livre
    grid[img_norm >= 0.5] = 0

    print(f"[INFO] Mapa carregado: {image_path}, tamanho = {grid.shape}")
    return grid


# === PRINT POLICY GRID ===
def print_policy(q_table, grid, goal_cell):
    # Símbolos para ações (ajusta conforme o teu Q-learning)
    action_symbols = ['^', '<', '>', 'o']  # forward, left, right, stay

    rows, cols = grid.shape
    policy_grid = [[' ' for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                policy_grid[r][c] = '#'
            elif (r, c) == goal_cell:
                policy_grid[r][c] = 'G'
            else:
                state = (r, c)
                if state in q_table and len(q_table[state]) > 0:
                    best_action = np.argmax(q_table[state])
                    policy_grid[r][c] = action_symbols[best_action]
                else:
                    policy_grid[r][c] = '.'

    print("\n=== POLÍTICA APRENDIDA ===")
    for row in policy_grid:
        print(" ".join(row))


# === MAIN ===
if __name__ == "__main__":
    # 1. Carregar Q-table
    q_table = load_q_table(QTABLE_PATH)
    if q_table is None:
        exit()

    # 2. Carregar mapa real
    grid = load_map_from_yaml(MAP_PATH)
    if grid is None:
        exit()

    # 3. Ajustar tamanho se necessário
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        print(f"[AVISO] Redimensionando mapa de {grid.shape} para ({GRID_SIZE}, {GRID_SIZE})")
        grid_img = Image.fromarray(grid.astype(np.uint8))
        grid_img = grid_img.resize((GRID_SIZE, GRID_SIZE), Image.NEAREST)
        grid = np.array(grid_img, dtype=int)

    # 4. Imprimir política aprendida
    print_policy(q_table, grid, GOAL_CELL)
