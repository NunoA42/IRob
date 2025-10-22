#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

# Caminhos dos ficheiros
BASE_DIR = "/root/catkin_ws/src/q_learning_robot"
SUMMARY_FILE = os.path.join(BASE_DIR, "episodes_summary.csv")
VISITS_FILE = os.path.join(BASE_DIR, "visit_counts.npy")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "q_snapshots")

def plot_curves():
    """Lê episodes_summary.csv e plota curvas de recompensa e taxa de sucesso."""
    episodes, rewards, success, success_rate = [], [], [], []
    if not os.path.exists(SUMMARY_FILE):
        print(f"❌ Arquivo {SUMMARY_FILE} não encontrado.")
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
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.title("Curva de Recompensa por Episódio")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, success_rate, label="Taxa de Sucesso (média móvel 100 ep)", color='green')
    plt.xlabel("Episódio")
    plt.ylabel("Taxa de Sucesso")
    plt.title("Curva da Taxa de Sucesso")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_visit_heatmap():
    """Mostra o mapa de calor das visitas."""
    if not os.path.exists(VISITS_FILE):
        print(f"❌ Arquivo {VISITS_FILE} não encontrado.")
        return
    
    visits = np.load(VISITS_FILE)
    visits_log = np.log1p(visits)  # escala logarítmica para visualização
    visits_norm = (visits_log / np.max(visits_log) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(visits_norm, cv2.COLORMAP_JET)

    # Permitir redimensionamento da janela
    cv2.namedWindow("Mapa de Visitas (Heatmap)", cv2.WINDOW_NORMAL)
    # Redimensionar para um tamanho maior (por exemplo, 800x600)
    cv2.resizeWindow("Mapa de Visitas (Heatmap)", 800, 600)

    cv2.imshow("Mapa de Visitas (Heatmap)", heatmap)
    print("🟩 Pressiona qualquer tecla para fechar o mapa...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_q_snapshots():
    """Mostra estatísticas da evolução do Q-learning a partir dos snapshots."""
    if not os.path.exists(SNAPSHOT_DIR):
        print("❌ Diretório de snapshots não encontrado.")
        return
    
    snapshot_files = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".pkl")])
    if not snapshot_files:
        print("⚠️ Nenhum snapshot encontrado.")
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
    plt.plot(episodes, avg_q_values, label="Q médio", linewidth=1.5)
    plt.plot(episodes, max_q_values, label="Q máximo", linewidth=1.5)
    plt.xlabel("Episódio (snapshot)")
    plt.ylabel("Valor de Q")
    plt.title("Evolução dos Q-values ao longo do treino")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("=== 🔍 Analisador de Q-Learning ROS ===")
    print("1️⃣ Curva de recompensa e sucesso")
    print("2️⃣ Mapa de calor (visitas)")
    print("3️⃣ Evolução do Q (snapshots)")
    print("Escolhe a opção (1/2/3 ou ENTER para todas): ", end="")
    choice = input().strip()

    if choice in ["1", ""]:
        plot_curves()
    if choice in ["2", ""]:
        plot_visit_heatmap()
    if choice in ["3", ""]:
        analyze_q_snapshots()
