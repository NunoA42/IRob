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
        # --- Parâmetros de Configuração ---
        self.GRID_SIZE = 52  # Dividir o mapa numa grelha de 32x32
        self.GOAL_CELL = (33, 36) # Célula objetivo (linha, coluna) - AJUSTE CONFORME O SEU MAPA!

        # Parâmetros do Q-Learning
        self.alpha = 0.2  # Taxa de aprendizagem
        self.gamma = 0.9  # Fator de desconto
        self.epsilon = 0.9 # Probabilidade de exploração inicial
        self.epsilon_decay = 0.99996 # Fator de decaimento para epsilon
        self.epsilon_min = 0.05   # Epsilon mínimo
        self.num_episodes = 100000

        # Definição de ações: 0: frente, 1: virar à esquerda, 2: virar à direita, 3: parar
        self.actions = ["forward", "turn_left", "turn_right", "stay"]
        self.num_actions = len(self.actions)
        self.num_orientations = 4  # 4 direções (N,E,S,W) — podes usar 8 para maior resolução


        # Recompensas
        self.goal_reward = 1000
        self.obstacle_penalty = -100
        self.step_penalty = -5

        # --- Variáveis de Estado ---
        self.q_table = {}  # Usar um dicionário para a Q-table: q_table[(state)] = [q_vals_for_actions]
        self.map_data = None
        self.map_info = None
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE)) # 0: livre, 1: obstáculo

        # Variáveis do robô simulado
        self.robot_pos = {'x': 0.0, 'y': 0.0}
        self.robot_theta = 0.0

        # Estatísticas por episódio
        self.episode_rewards = []
        self.episode_success = []

        # Contador de visitas ao mapa
        self.visit_counts = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)

        # Diretório para snapshots
        self.snapshot_dir = "/root/catkin_ws/src/q_learning_robot/q_snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)


        # --- Inicialização do ROS ---
        rospy.init_node('q_learning_node')

        # Subscribers e Publishers
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("Nó de Q-learning iniciado. À espera do mapa...")
        rospy.spin()

    def map_callback(self, msg):
        if self.map_data is not None:
            return # Processar o mapa apenas uma vez

        rospy.loginfo("Mapa recebido!")
        self.map_data = msg.data
        self.map_info = msg.info
        
        self.process_map()
        self.run_q_learning()

    def process_map(self):
        """ Converte o OccupancyGrid do ROS para a nossa grelha interna 32x32. """
        map_width = self.map_info.width
        map_height = self.map_info.height

        # Calcular o tamanho de cada célula da nossa grelha no mapa original
        cell_width_map = map_width / self.GRID_SIZE
        cell_height_map = map_height / self.GRID_SIZE

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                is_obstacle = False
                # Verificar uma área no mapa original correspondente a esta célula da grelha
                for i in range(int(r * cell_height_map), int((r + 1) * cell_height_map)):
                    for j in range(int(c * cell_width_map), int((c + 1) * cell_width_map)):
                        map_index = i * map_width + j
                        if self.map_data[map_index] > 99:  # limiar de obstáculo
                            is_obstacle = True
                            break
                    if is_obstacle:
                        break
                
                if is_obstacle:
                    self.grid[r, c] = 1 # Obstáculo

        # Assegurar que a célula objetivo não é um obstáculo
        if self.grid[self.GOAL_CELL] == 1:
            self.grid[self.GOAL_CELL] = 0
            rospy.logwarn("A célula objetivo estava marcada como obstáculo. Foi limpa.")
        
                # Definição dos limites no mundo
        limite_x_min = -0.9
        limite_x_max = 5.5
        limite_y_min = -3.1
        limite_y_max = 2.9

        # Dimensões das células no mundo
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

        # Definição dos dois retângulos
        # Retângulo 1
        x1_min, x1_max = -1.5, 0.7
        y1_min, y1_max = 1.85, 4.3

        # Retângulo 2
        x2_min, x2_max = 0.14, 3.9
        y2_min, y2_max = -1.3, -0.14

        # Retângulo 3
        x3_min, x3_max = -0.61, 4.19
        y3_min, y3_max = 0.54, 1.66

        # Retângulo 4 (novo)
        x4_min, x4_max = -1.3170, 4.4620
        y4_min, y4_max = 0.5298, 2.5689

        # Dimensões reais de cada célula da grelha
        cell_width_m = (self.map_info.width / self.GRID_SIZE) * self.map_info.resolution
        cell_height_m = (self.map_info.height / self.GRID_SIZE) * self.map_info.resolution

        # Marcar obstáculos dentro destes retângulos
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                x_celula = c * cell_width_m + self.map_info.origin.position.x
                y_celula = r * cell_height_m + self.map_info.origin.position.y

                if (
                    (x1_min <= x_celula <= x1_max and y1_min <= y_celula <= y1_max) or
                    (x2_min <= x_celula <= x2_max and y2_min <= y_celula <= y2_max) or
                    (x3_min <= x_celula <= x3_max and y3_min <= y_celula <= y3_max) or
                    (x4_min <= x_celula <= x4_max and y4_min <= y_celula <= y4_max)
                ):
                    self.grid[r, c] = 1

        # --- Forçar células específicas a serem livres ---
        coordenadas_livres = [
            (1.621452808380127, -2.0439205169677734),
            (2.0016815662384033, -2.0534446239471436)
        ]

        for (x, y) in coordenadas_livres:
            # Converter coordenadas do mundo para índice de grelha
            c = int((x - self.map_info.origin.position.x) / self.map_info.resolution / (self.map_info.width / self.GRID_SIZE))
            r = int((y - self.map_info.origin.position.y) / self.map_info.resolution / (self.map_info.height / self.GRID_SIZE))
            
            # Manter dentro dos limites
            r = max(0, min(self.GRID_SIZE - 1, r))
            c = max(0, min(self.GRID_SIZE - 1, c))
            
            # Marcar como livre
            self.grid[r, c] = 0

        rospy.loginfo("Mapa processado para uma grelha de %dx%d.", self.GRID_SIZE, self.GRID_SIZE)

    def get_orientation_index(self):
        """Discretiza robot_theta em self.num_orientations buckets."""
        theta = self.robot_theta  # -pi..pi
        frac = (theta + math.pi) / (2 * math.pi)  # 0..1
        idx = int(round(frac * self.num_orientations)) % self.num_orientations
        return idx

    def get_state_from_pos(self, x, y):
        """Retorna (r, c, orient_idx)."""
        grid_c = int((x - self.map_info.origin.position.x) / self.map_info.resolution / (self.map_info.width / self.GRID_SIZE))
        grid_r = int((y - self.map_info.origin.position.y) / self.map_info.resolution / (self.map_info.height / self.GRID_SIZE))
        grid_r = max(0, min(self.GRID_SIZE - 1, grid_r))
        grid_c = max(0, min(self.GRID_SIZE - 1, grid_c))
        orient = self.get_orientation_index()
        return (grid_r, grid_c, orient)     

    def get_world_coords_from_state(self, state):
        """ Converte o estado da grelha (linha, coluna[, orient]) para coordenadas do mundo (x, y). """
        # aceita state com 2 ou 3 elementos
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
        """ Encontra uma célula livre aleatória para iniciar um episódio. """
        while True:
            r,c = (23, 39)
            r, c = np.random.randint(0, self.GRID_SIZE, size=2)
            if self.grid[r, c] == 0:
                return (r, c)

    def choose_action(self, state):
        """ Escolhe uma ação usando a política epsilon-greedy. """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions) # Ação aleatória (exploração)
        else:
            q_values = self.q_table.get(state, np.zeros(self.num_actions))
            return np.argmax(q_values) # Melhor ação (exploração)

    def step(self, state, action_idx):
        """ Simula um passo do robô e retorna o novo estado, a recompensa e se o episódio terminou. """
        action = self.actions[action_idx]
        
        # --- Modelo de movimento simples ---
        step_dist = 0.2  # metros
        turn_angle = math.pi / 2  # 90 graus

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
        # "stay" não altera posição

        # Normalizar o ângulo
        self.robot_theta = math.atan2(math.sin(self.robot_theta), math.cos(self.robot_theta))
        
        # Obter novo estado
        new_state = self.get_state_from_pos(self.robot_pos['x'], self.robot_pos['y'])
        
        done = False
        reward = self.step_penalty

        # --- 🔹 Reward shaping potencial ---
        goal_x, goal_y = self.get_world_coords_from_state(self.GOAL_CELL)
        old_x, old_y = self.get_world_coords_from_state(state)
        old_dist = math.hypot(goal_x - old_x, goal_y - old_y)
        new_dist = math.hypot(goal_x - self.robot_pos['x'], goal_y - self.robot_pos['y'])

        phi_old = -old_dist
        #phi_new = -new_dist

        # potencial-based reward shaping (teoricamente correto)
        #shaping_reward = self.gamma * phi_new - phi_old

        # Escala opcional (podes ajustar)
        #reward += 5.0 * shaping_reward

        if action == "stay":
            reward -= 50

        # --- Verificar fim do episódio ---
        # compara apenas (r,c) com GOAL_CELL
        if (new_state[0], new_state[1]) == self.GOAL_CELL:
            reward = self.goal_reward
            done = True
        elif self.grid[new_state[0], new_state[1]] == 1:
            reward = self.obstacle_penalty
            done = True   # mantém o comportamento original (termina no obstáculo)

            # Voltar à posição anterior (comentado no original)
            #self.robot_pos['x'], self.robot_pos['y'] = self.get_world_coords_from_state(state)
            #new_state = state

        return new_state, reward, done


    def run_q_learning(self):
        """ Loop principal de treino do Q-learning. """
        rospy.loginfo("A iniciar treino de Q-learning...")
        
        self.q_value_progress = []
        self.policy_snapshots = []

        # --- Carregar Q-table existente, se disponível ---
        self.load_q_table()

        for episode in range(self.num_episodes):
            initial_state = self.get_random_free_state()
            # posiciona o robô no mundo de acordo com a célula inicial (r,c)
            self.robot_pos['x'], self.robot_pos['y'] = self.get_world_coords_from_state(initial_state)
            # mantém theta (podes randomizar se quiseres)
            #self.robot_theta = np.random.uniform(-math.pi, math.pi)

            # agora obtém o estado completo (r,c,orient)
            state = self.get_state_from_pos(self.robot_pos['x'], self.robot_pos['y'])

            # Track initial Q-value stats each episode
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

                # Contar visita
                r, c = state[0], state[1]
                self.visit_counts[r, c] += 1

                # Publicar para RViz
                #self.publish_visualization(state)
                #self.publish_tf()
                #rospy.sleep(0.01)

                if done:
                    if (next_state[0], next_state[1]) == self.GOAL_CELL:
                        success = True  # 🔹 ADIÇÃO — marcou sucesso
                    break


            # Decaimento do epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Guardar resultados do episódio
            self.episode_rewards.append(total_reward)
            self.episode_success.append(success)

            # 🔹 ADIÇÃO — Guardar snapshot da Q-table a cada N episódios
            if (episode + 1) % 5000 == 0:
                snapshot_path = os.path.join(self.snapshot_dir, f"q_snapshot_ep{episode+1}.pkl")
                with open(snapshot_path, "wb") as f:
                    pickle.dump(self.q_table, f)


            # --- Log e Salvamento periódico ---
            if (episode + 1) % 10000 == 0:  # muda aqui o número conforme quiser
                rospy.loginfo("Episódio %d/%d concluído. Recompensa total: %.2f. Epsilon: %.3f", 
                              episode + 1, self.num_episodes, total_reward, self.epsilon)
                self.save_q_table()  # guarda o progresso automaticamente
                # Save a snapshot of the learned policy every few episodes
                self.policy_snapshots.append((episode + 1, self.extract_policy_grid()))


        rospy.loginfo("Treino concluído.")
        self.save_q_table()  # salva também no final, por segurança
        rospy.loginfo("Q-table final guardada.")
        self.print_policy()
        self.publish_visualization(self.GOAL_CELL)
         # --- Save Q-value and policy evolution ---
        self.save_progress_to_csv()
        self.save_episode_summary()
        self.save_visit_map()



    def print_policy(self):
        """ Imprime a política aprendida no terminal. """
        rospy.loginfo("--- Política Ótima Aprendida ---")
        policy_grid = [[' ' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 1:
                    policy_grid[r][c] = '#'
                elif (r, c) == self.GOAL_CELL:
                    policy_grid[r][c] = 'G'
                else:
                    # Agrega as orientações — escolhe melhor ação considerando todas as orientações
                    best_action_idx = None
                    # construir um vetor agregado (máx sobre orientações)
                    aggregated = np.zeros(self.num_actions)
                    found_any = False
                    for o in range(self.num_orientations):
                        st = (r, c, o)
                        if st in self.q_table:
                            aggregated = np.maximum(aggregated, self.q_table[st])
                            found_any = True
                    if found_any:
                        best_action_idx = np.argmax(aggregated)
                        action_symbols = ['^', '<', '>', 'o'] # Frente, Esquerda, Direita, Parar
                        policy_grid[r][c] = action_symbols[best_action_idx]
        
        # Imprime a grelha da política
        for row in policy_grid:
            print(" ".join(row))
        rospy.loginfo("-----------------------------")

    
    def save_q_table(self, filename="/root/catkin_ws/src/q_learning_robot/q_table.pkl"):
        """Guarda a Q-table e parâmetros de treino num ficheiro."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        rospy.loginfo("Q-table guardada em %s", filename)

    def load_q_table(self, filename="/root/catkin_ws/src/q_learning_robot/q_table.pkl"):
        """Carrega a Q-table e parâmetros de treino de um ficheiro, se existir."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon)
                self.alpha = data.get('alpha', self.alpha)
                self.gamma = data.get('gamma', self.gamma)
            rospy.loginfo("Q-table carregada de %s", filename)
        else:
            rospy.logwarn("Nenhuma Q-table encontrada em %s. A iniciar do zero.", filename)


    def publish_visualization(self, robot_current_state):
        """ Publica a grelha colorida como um MarkerArray no RViz. """
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

                # Posição do centro da célula
                x, y = self.get_world_coords_from_state((r, c))
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = -0.01 # Ligeiramente abaixo do mapa para não o tapar

                # Escala da célula
                cell_width_world = (self.map_info.width / self.GRID_SIZE) * self.map_info.resolution
                cell_height_world = (self.map_info.height / self.GRID_SIZE) * self.map_info.resolution
                marker.scale.x = cell_width_world
                marker.scale.y = cell_height_world
                marker.scale.z = 0.01

                marker.color.a = 0.7 # Transparência

                # Cores
                if self.grid[r, c] == 1: # Obstáculo
                    marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0) # Vermelho
                elif (r, c) == self.GOAL_CELL: # Objetivo
                    marker.color.r, marker.color.g, marker.color.b = (0.0, 1.0, 0.0) # Verde
                else:
                    # robot_current_state pode ser (r,c) ou (r,c,orient)
                    if isinstance(robot_current_state, tuple) and len(robot_current_state) >= 2 and (r, c) == (robot_current_state[0], robot_current_state[1]):
                        marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.0) # Amarelo
                    else:
                        marker.color.r, marker.color.g, marker.color.b = (0.0, 0.0, 1.0) # Azul

                marker_array.markers.append(marker)
                marker_id += 1
        
        self.marker_pub.publish(marker_array)

    def publish_tf(self):
        """ Publica a transformação da pose simulada do robô. """
        self.tf_broadcaster.sendTransform(
            (self.robot_pos['x'], self.robot_pos['y'], 0),
            tf.transformations.quaternion_from_euler(0, 0, self.robot_theta),
            rospy.Time.now(),
            "base_link",
            "odom"
        )

    def extract_policy_grid(self):
        """Return the learned policy grid (symbols) for saving or analysis."""
        policy_grid = [[' ' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 1:
                    policy_grid[r][c] = '#'
                elif (r, c) == self.GOAL_CELL:
                    policy_grid[r][c] = 'G'
                else:
                    # Agrega orientações para determinar a ação representativa
                    aggregated = np.zeros(self.num_actions)
                    found_any = False
                    for o in range(self.num_orientations):
                        st = (r, c, o)
                        if st in self.q_table:
                            aggregated = np.maximum(aggregated, self.q_table[st])
                            found_any = True
                    if found_any:
                        best_action_idx = np.argmax(aggregated)
                        symbols = ['^', '<', '>', 'o']  # forward, left, right, stay
                        policy_grid[r][c] = symbols[best_action_idx]
        return policy_grid


    def save_progress_to_csv(self, q_values_file="/root/catkin_ws/src/q_learning_robot/q_values_progress.csv",
                             policies_file="/root/catkin_ws/src/q_learning_robot/policies_progress.csv"):
        """Save Q-values evolution and policy snapshots to CSV files."""
        import csv

        # --- Save Q-value statistics ---
        with open(q_values_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Average_Q_Value", "Max_Q_Value"])
            for ep, avg_q, max_q in self.q_value_progress:
                writer.writerow([ep, avg_q, max_q])

        # --- Save policy snapshots ---
        with open(policies_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Policy_Grid"])
            for ep, grid in self.policy_snapshots:
                grid_flat = ["".join(row) for row in grid]
                writer.writerow([ep, "|".join(grid_flat)])

        rospy.loginfo("Progresso de Q-values e políticas guardado em CSV.")


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
        """Guarda recompensa média e taxa de sucesso por episódio."""
        import csv
        success_rate = []
        window = 100  # média móvel dos últimos 100 episódios
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
        
        rospy.loginfo("Resumo de episódios guardado em %s", filename)


    def save_visit_map(self, filename="/root/catkin_ws/src/q_learning_robot/visit_counts.npy"):
        """Guarda o mapa de contagens de visitas."""
        np.save(filename, self.visit_counts)
        rospy.loginfo("Mapa de visitas guardado em %s", filename)


if __name__ == '__main__':
    try:
        QLearningROS()
    except rospy.ROSInterruptException:
        pass
