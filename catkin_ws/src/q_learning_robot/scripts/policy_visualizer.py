#!/usr/bin/env python3

import numpy as np
import pickle
import sys

class TerminalPolicyVisualizer:
    def __init__(self):
        # Parâmetros (devem corresponder aos do treino)
        self.GRID_SIZE = 52
        self.GOAL_CELL = (33, 36)
        self.num_orientations = 4  # N, E, S, W
        self.actions = ["forward", "turn_left", "turn_right", "stay"]
        
        # Símbolos para as setas
        self.arrow_symbols = {
            'N': '↑',  # Norte
            'E': '→',  # Este
            'S': '↓',  # Sul
            'W': '←',  # Oeste
            'NE': '↗',
            'SE': '↘',
            'SW': '↙',
            'NW': '↖',
            'GOAL': '★',
            'EMPTY': '·',
            'NO_DATA': '░'
        }
        
        # Cores ANSI
        self.colors = {
            'reset': '\033[0m',
            'goal': '\033[92m',      # Verde brilhante
            'high': '\033[96m',       # Ciano
            'medium': '\033[94m',     # Azul
            'low': '\033[90m',        # Cinza
            'no_data': '\033[2m'      # Dim
        }
        
        self.q_table = {}
        self.load_q_table()
    
    def load_q_table(self, filename="/root/catkin_ws/src/q_learning_robot/q_table.pkl"):
        """Carrega a Q-table do ficheiro."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
            print(f"✓ Q-table carregada: {len(self.q_table)} estados")
        except FileNotFoundError:
            print(f"✗ Erro: Ficheiro não encontrado: {filename}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Erro ao carregar Q-table: {str(e)}")
            sys.exit(1)
    
    def get_orientation_name(self, orient_idx):
        """Converte índice de orientação para nome."""
        orientations = ['E', 'S', 'W', 'N']  # 0: Este, 1: Sul, 2: Oeste, 3: Norte (corrigido para Y invertido)
        return orientations[orient_idx % self.num_orientations]
    
    def get_action_result_orientation(self, base_orient, action_idx):
        """Calcula a orientação resultante após a ação."""
        if action_idx == 0:  # forward - mantém orientação
            return base_orient
        elif action_idx == 1:  # turn_left
            orientations = ['E', 'S', 'W', 'N']
            current_name = self.get_orientation_name(base_orient)
            current_idx = orientations.index(current_name)
            new_idx = (current_idx + 1) % 4  # Rodar anti-horário
            return new_idx
        elif action_idx == 2:  # turn_right
            orientations = ['E', 'S', 'W', 'N']
            current_name = self.get_orientation_name(base_orient)
            current_idx = orientations.index(current_name)
            new_idx = (current_idx - 1) % 4  # Rodar horário
            return new_idx
        else:  # stay
            return None
    
    def get_best_action_for_cell(self, r, c):
        """Retorna a melhor ação agregada para uma célula."""
        all_actions = []
        
        for orient_idx in range(self.num_orientations):
            state = (r, c, orient_idx)
            if state in self.q_table:
                q_values = self.q_table[state]
                best_action = np.argmax(q_values)
                max_q = q_values[best_action]
                
                if max_q > -900:  # Ignorar se não aprendeu
                    all_actions.append((best_action, orient_idx, max_q))
        
        if not all_actions:
            return None, None, None
        
        # Encontrar a ação com maior Q-value
        best_entry = max(all_actions, key=lambda x: x[2])
        best_action, best_orient, max_q = best_entry
        
        return best_action, best_orient, max_q
    
    def get_arrow_for_cell(self, r, c):
        """Retorna o símbolo da seta e cor para uma célula."""
        # Célula objetivo
        if (r, c) == self.GOAL_CELL:
            return self.arrow_symbols['GOAL'], self.colors['goal'], 1.0
        
        # Obter melhor ação
        best_action, best_orient, max_q = self.get_best_action_for_cell(r, c)
        
        if best_action is None:
            return self.arrow_symbols['NO_DATA'], self.colors['no_data'], 0.0
        
        if best_action == 3:  # stay
            return self.arrow_symbols['EMPTY'], self.colors['low'], 0.0
        
        # Calcular orientação resultante
        result_orient = self.get_action_result_orientation(best_orient, best_action)
        
        if result_orient is None:
            return self.arrow_symbols['EMPTY'], self.colors['low'], 0.0
        
        # Obter símbolo da seta
        orient_name = self.get_orientation_name(result_orient)
        symbol = self.arrow_symbols.get(orient_name, self.arrow_symbols['EMPTY'])
        
        # Determinar cor baseada no Q-value
        q_normalized = min(1.0, max(0.0, (max_q + 1000) / 2000))
        
        if q_normalized > 0.7:
            color = self.colors['high']
        elif q_normalized > 0.4:
            color = self.colors['medium']
        else:
            color = self.colors['low']
        
        return symbol, color, q_normalized
    
    def print_policy_grid(self, scale=1):
        """Imprime a grelha da política no terminal."""
        print("\n" + "="*60)
        print("  POLÍTICA APRENDIDA (Q-Learning)")
        print("="*60)
        print(f"\n  {self.colors['goal']}★{self.colors['reset']} = Objetivo")
        print(f"  {self.colors['high']}↑→↓←{self.colors['reset']} = Alta confiança (Q-value alto)")
        print(f"  {self.colors['medium']}↑→↓←{self.colors['reset']} = Média confiança")
        print(f"  {self.colors['low']}↑→↓←{self.colors['reset']} = Baixa confiança")
        print(f"  {self.colors['no_data']}░{self.colors['reset']} = Sem dados\n")
        
        # Imprimir a grelha
        step = max(1, self.GRID_SIZE // (50 * scale))  # Ajustar densidade
        
        print("    ", end="")
        for c in range(0, self.GRID_SIZE, step):
            print(f"{c:2d} ", end="")
        print()
        
        for r in range(0, self.GRID_SIZE, step):
            print(f"{r:2d}  ", end="")
            for c in range(0, self.GRID_SIZE, step):
                symbol, color, _ = self.get_arrow_for_cell(r, c)
                print(f"{color}{symbol}{self.colors['reset']}  ", end="")
            print()
        
        print("\n" + "="*60 + "\n")
    
    def print_statistics(self):
        """Imprime estatísticas sobre a política."""
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        cells_with_data = 0
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) == self.GOAL_CELL:
                    continue
                
                _, _, q_norm = self.get_arrow_for_cell(r, c)
                
                if q_norm > 0:
                    cells_with_data += 1
                    if q_norm > 0.7:
                        high_confidence += 1
                    elif q_norm > 0.4:
                        medium_confidence += 1
                    else:
                        low_confidence += 1
        
        print("ESTATÍSTICAS DA POLÍTICA:")
        print("-" * 40)
        print(f"  Células totais:          {total_cells}")
        print(f"  Células com política:    {cells_with_data} ({100*cells_with_data/total_cells:.1f}%)")
        print(f"  Alta confiança:          {high_confidence} ({100*high_confidence/cells_with_data:.1f}%)")
        print(f"  Média confiança:         {medium_confidence} ({100*medium_confidence/cells_with_data:.1f}%)")
        print(f"  Baixa confiança:         {low_confidence} ({100*low_confidence/cells_with_data:.1f}%)")
        print("-" * 40 + "\n")
    
    def visualize(self, scale=1):
        """Visualiza a política no terminal."""
        self.print_policy_grid(scale)
        self.print_statistics()

def main():
    # Verificar argumentos
    if len(sys.argv) > 1:
        q_table_path = sys.argv[1]
    else:
        q_table_path = "/root/catkin_ws/src/q_learning_robot/q_table.pkl"
    
    # Criar visualizador
    viz = TerminalPolicyVisualizer()
    viz.q_table = {}
    
    # Carregar Q-table do caminho especificado
    try:
        with open(q_table_path, 'rb') as f:
            data = pickle.load(f)
            viz.q_table = data.get('q_table', {})
        print(f"✓ Q-table carregada: {len(viz.q_table)} estados")
    except Exception as e:
        print(f"✗ Erro: {str(e)}")
        sys.exit(1)
    
    # Visualizar
    viz.visualize(scale=1)

if __name__ == '__main__':
    main()