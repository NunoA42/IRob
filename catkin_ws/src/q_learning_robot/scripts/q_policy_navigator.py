#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Policy Navigator for TurtleBot3 (ROS1)
--------------------------------------

• Lê uma Q-table exportada (pickle) do teu agente discreto.
• Converte célula (r,c) -> alvo em coordenadas do mundo (map frame).
• Envia objetivos para o move_base (Navigation Stack) célula a célula.
• Pode receber o "próximo quadrado" externamente via tópico /q_target_cell
  (std_msgs/Int32MultiArray com [row, col]). Se não receber, decide pelo
  melhor passo a partir da Q-table (greedy) face ao estado atual.

Requisitos:
  - amcl já a publicar pose no /amcl_pose
  - map_server a publicar /map
  - move_base a correr e com costmaps configurados
  - ficheiro pickle da Q-table com o formato usado no teu treino:
      {
        'q_table': { (r, c, o): np.array([q_fwd, q_left, q_right, q_stay]), ... },
        'epsilon': float, 'alpha': float, 'gamma': float
      }

Executar:
  rosrun <seu_pacote> q_policy_navigator.py _q_table_path:=/caminho/q_table.pkl _grid_size:=52

Tópicos:
  Sub: /map (nav_msgs/OccupancyGrid)
       /amcl_pose (geometry_msgs/PoseWithCovarianceStamped)
       /q_target_cell (std_msgs/Int32MultiArray)  # opcional
  Pub: /q_decided_cell (std_msgs/Int32MultiArray) # só para debug/visualização

"""
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
        # --- Parâmetros ROS ---
        self.q_table_path = rospy.get_param('~q_table_path', '/root/catkin_ws/src/q_learning_robot/q_table.pkl')
        self.grid_size = int(rospy.get_param('~grid_size', 52))
        self.num_orientations = int(rospy.get_param('~num_orientations', 4))  # deve coincidir com o treino
        self.goal_timeout = float(rospy.get_param('~goal_timeout', 60.0))     # s por célula
        self.retry_limit = int(rospy.get_param('~retry_limit', 2))
        self.frame_map = rospy.get_param('~frame_map', 'map')
        # Tolerâncias (posição em metros; ângulo em graus)
        self.pos_tolerance = float(rospy.get_param('~pos_tolerance', 0.15))
        self.yaw_tolerance_deg = float(rospy.get_param('~yaw_tolerance_deg', 15.0))
        self.yaw_tolerance = math.radians(self.yaw_tolerance_deg)

        # --- Estado do mapa/pose ---
        self.map_info = None  # guardará msg.info do OccupancyGrid
        self.have_map = False
        self.pose_xytheta = None  # (x, y, yaw)

        # --- Q-table ---
        self.q_table = {}
        self.actions = ['forward', 'turn_left', 'turn_right', 'stay']
        self.load_q_table(self.q_table_path)

        # --- ROS I/O ---
        rospy.Subscriber('/map', OccupancyGrid, self.cb_map, queue_size=1)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.cb_pose, queue_size=10)
        rospy.Subscriber('/q_target_cell', Int32MultiArray, self.cb_external_target, queue_size=10)
        self.decided_pub = rospy.Publisher('/q_decided_cell', Int32MultiArray, queue_size=10)

        # --- Cliente action para move_base ---
        self.mb_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("[QPolicyNavigator] À espera do servidor move_base...")
        self.mb_client.wait_for_server()
        rospy.loginfo("[QPolicyNavigator] Conectado ao move_base.")

        # Timer principal: se não chega alvo externo, segue política Q-table
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_step)

        # Cache último alvo decidido (para evitar spam de objetivos idênticos)
        self.last_target_cell = None

    # ---------------------- Callbacks ----------------------
    def cb_map(self, msg: OccupancyGrid):
        if not self.have_map:
            self.map_info = msg.info
            self.have_map = True
            rospy.loginfo("[QPolicyNavigator] /map recebido: %dx%d, res=%.3fm",
                          msg.info.width, msg.info.height, msg.info.resolution)

    def cb_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose_xytheta = (p.x, p.y, self.norm_angle(yaw))

    def cb_external_target(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            rospy.logwarn("[QPolicyNavigator] /q_target_cell inválido (precisa de [row, col]).")
            return
        r, c = int(msg.data[0]), int(msg.data[1])
        rospy.loginfo("[QPolicyNavigator] Alvo externo recebido: (%d,%d)", r, c)
        self.navigate_to_cell((r, c))

    # ---------------------- Timer loop ---------------------
    def timer_step(self, _event):
        # Só decide automaticamente se não há alvo externo recente
        if not (self.have_map and self.pose_xytheta and self.q_table):
            return

        # Estado atual -> célula atual e orientação discreta
        cur_cell = self.world_to_cell(self.pose_xytheta[0], self.pose_xytheta[1])
        orient_idx = self.get_orientation_index(self.pose_xytheta[2])
        state = (cur_cell[0], cur_cell[1], orient_idx)

        # Se não temos Qs para este estado, não decide
        if state not in self.q_table:
            return

        # Escolhe ação greedy
        qvals = self.q_table[state]
        action_idx = int(np.argmax(qvals))
        action = self.actions[action_idx]

        # Determina célula seguinte, alinhando com a direção (após turn, vai 1 célula em frente)
        desired_yaw = self.pose_xytheta[2]
        if action == 'turn_left':
            desired_yaw = self.norm_angle(desired_yaw + math.pi/2)
        elif action == 'turn_right':
            desired_yaw = self.norm_angle(desired_yaw - math.pi/2)
        # 'forward' mantém yaw; 'stay' mantém célula

        next_cell = cur_cell if action == 'stay' else self.pick_best_neighbor_towards_yaw(cur_cell, desired_yaw)

        if next_cell is None:
            return

        # Publica para debug
        out = Int32MultiArray(data=[int(next_cell[0]), int(next_cell[1])])
        self.decided_pub.publish(out)

        # Evita reenviar o mesmo objetivo repetidamente
        if self.last_target_cell != next_cell:
            rospy.loginfo("[QPolicyNavigator] Ação=%s -> célula seguinte %s", action, next_cell)
            self.navigate_to_cell(next_cell)
            self.last_target_cell = next_cell

    # ---------------------- Navegação ----------------------
    def navigate_to_cell(self, cell_rc):
        if not self.have_map or self.pose_xytheta is None:
            rospy.logwarn("[QPolicyNavigator] À espera de /map e /amcl_pose para navegar...")
            return

        r, c = int(cell_rc[0]), int(cell_rc[1])
        goal_xy = self.cell_to_world_center(r, c)

        # yaw a apontar ao centro do alvo
        dx = goal_xy[0] - self.pose_xytheta[0]
        dy = goal_xy[1] - self.pose_xytheta[1]
        yaw = math.atan2(dy, dx)

        # ✅ Se já estamos dentro das tolerâncias, considera atingido sem enviar goal
        if self.within_tolerance(goal_xy[0], goal_xy[1], yaw):
            rospy.loginfo("[QPolicyNavigator] Já dentro da tolerância da célula (%d,%d).", r, c)
            return

        # Envia objetivo ao move_base com verificação contínua de tolerâncias
        for attempt in range(self.retry_limit + 1):
            ok = self.send_move_base_goal(goal_xy[0], goal_xy[1], yaw)
            if ok:
                rospy.loginfo("[QPolicyNavigator] Alcançou célula (%d,%d).", r, c)
                return
            else:
                rospy.logwarn("[QPolicyNavigator] Falhou célula (%d,%d), tentativa %d/%d.",
                              r, c, attempt + 1, self.retry_limit + 1)
        rospy.logerr("[QPolicyNavigator] Desisti de alcançar célula (%d,%d).", r, c)

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

        # Se já dentro da tolerância, retorna sucesso imediato
        if self.within_tolerance(x, y, yaw):
            return True

        self.mb_client.send_goal(goal)

        # Enquanto o objetivo está ativo, monitoriza pose e aplica tolerâncias
        rate = rospy.Rate(10)  # 10 Hz
        start = rospy.Time.now()
        while (rospy.Time.now() - start).to_sec() < self.goal_timeout:
            if self.within_tolerance(x, y, yaw):
                # Dentro da tolerância — cancela e considera sucesso
                try:
                    self.mb_client.cancel_goal()
                except Exception:
                    pass
                return True
            # Se o actionlib reportar sucesso antes, aceita também
            state = self.mb_client.get_state()
            if state == 3:
                return True
            rate.sleep()

        # Timeout — cancela e falha
        rospy.logwarn("[QPolicyNavigator] Timeout do objetivo (%.1fs). A cancelar...", self.goal_timeout)
        self.mb_client.cancel_goal()
        return False

    def within_tolerance(self, gx, gy, gyaw):
        if self.pose_xytheta is None:
            return False
        x, y, yaw = self.pose_xytheta
        dist = math.hypot(gx - x, gy - y)
        dyaw = self.norm_angle(gyaw - yaw)
        return (dist <= self.pos_tolerance) and (abs(dyaw) <= self.yaw_tolerance)

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

        self.mb_client.send_goal(goal)
        finished = self.mb_client.wait_for_result(rospy.Duration(self.goal_timeout))
        if not finished:
            rospy.logwarn("[QPolicyNavigator] Timeout do objetivo (%.1fs). A cancelar...", self.goal_timeout)
            self.mb_client.cancel_goal()
            return False
        state = self.mb_client.get_state()
        # 3 == SUCCEEDED em actionlib_msgs/GoalStatus
        return state == 3

    # ---------------------- Utilitários de grelha ----------------------
    def cell_to_world_center(self, row, col):
        """Centro da célula (r,c) em coordenadas do mundo (map)."""
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
        """Escolhe o vizinho 4-conexo cujo vetor ao centro maximiza o alinhamento com desired_yaw."""
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
        """Discretiza yaw em num_orientations buckets (compatível com o treino)."""
        frac = (theta + math.pi) / (2 * math.pi)  # 0..1
        idx = int(round(frac * self.num_orientations)) % self.num_orientations
        return idx

    @staticmethod
    def norm_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    # ---------------------- Q-table I/O --------------------
    def load_q_table(self, path):
        if not os.path.exists(path):
            rospy.logwarn("[QPolicyNavigator] Q-table não encontrada em %s — modo apenas com alvo externo.", path)
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data.get('q_table', {})
        rospy.loginfo("[QPolicyNavigator] Q-table carregada (%d estados).", len(self.q_table))


def main():
    rospy.init_node('q_policy_navigator')
    node = QPolicyNavigator()
    rospy.loginfo("[QPolicyNavigator] Pronto. Publica /q_target_cell para forçar o próximo quadrado ou deixa-o seguir a política.")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass