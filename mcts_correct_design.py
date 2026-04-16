#!/usr/bin/env python3
"""
MCTS CORREGIDO - Reproducer acciones desde raíz en lugar de guardar/restaurar estado

El problema en mcts_with_tensorboard.py:
- Solo guardaba estado en raíz en cada iteración
- Cuando descendía por hijos, NO restauraba su estado
- Causaba desincronización entre state guardado e iterations reales

Solución:
- En lugar de guardar state en cada nodo, guarda las ACCIONES en la ruta
- En cada iteración, reproduce las acciones desde raíz
- Garantiza consistencia del environment
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '/home/servergmun/MCTS_CUCA_CACA')

from Environment import BinaryMathEnvCUDA


class MCTSNodeCorrect:
    """MCTS node que NO guarda state, solo acciones"""

    def __init__(self, parent=None, action=None, depth=0):
        self.parent = parent
        self.action = action  # Acción que llevó a este nodo
        self.depth = depth

        self.children = {}  # action_idx → MCTSNode
        self.visits = 0
        self.reward_sum = 0.0
        self.untried_actions = None  # Se inicializa cuando se accede

    def init_untried_actions(self, n_actions):
        """Inicializa acciones sin probar (solo una vez)"""
        if self.untried_actions is None:
            self.untried_actions = list(range(n_actions))
            random.shuffle(self.untried_actions)

    def get_action_sequence(self):
        """Devuelve la secuencia de acciones desde raíz hasta este nodo"""
        sequence = []
        node = self
        while node.parent is not None:
            sequence.append(node.action)
            node = node.parent
        return sequence[::-1]  # Invertir para obtener orden desde raíz

    def uct_value(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        exploit = self.reward_sum / self.visits
        if self.parent is None or self.parent.visits == 0:
            return exploit
        explore = c * np.sqrt(np.log(self.parent.visits) / (1 + self.visits))
        return exploit + explore

    def best_child(self, c=1.41):
        if not self.children:
            return None
        return max(self.children.values(), key=lambda ch: ch.uct_value(c))

    def is_fully_expanded(self):
        """¿Están TODAS las acciones exploradas?

        MCTS correcto: solo permite descender si esta propiedad es True.
        Así se fuerza exploración lateral antes de permitir profundidad.
        """
        return len(self.untried_actions) == 0

    def update(self, reward):
        self.visits += 1
        self.reward_sum += reward

    def avg_reward(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0


class MCTSCorrectDesign:
    def __init__(self, env, n_iterations=2000, c=2.0, log_dir='./runs'):
        self.env = env
        self.n_iterations = n_iterations
        self.c = c
        self.root = MCTSNodeCorrect(parent=None, action=None, depth=0)
        self.root.init_untried_actions(env.n_actions)

        # TensorBoard
        self.log_dir = log_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"mcts_correct_bits{env.Bits}_c{c:.2f}_{timestamp}"
        self.writer = SummaryWriter(f'{log_dir}/{run_name}')
        print(f"\n📊 TensorBoard: tensorboard --logdir {log_dir}")
        print(f"📊 Run: {run_name}\n")

        # Metrics
        self.iteration_rewards = []
        self.best_rewards = []
        self.tree_stats = {
            'total_nodes': [],
            'max_depth': [],
            'avg_depth': [],
            'root_children': [],
            'branching_factor': [],
        }

        self.best_policy = None
        self.best_reward_found = -float('inf')

    def count_nodes(self, node):
        return 1 + sum(self.count_nodes(child) for child in node.children.values())

    def max_depth(self, node):
        if not node.children:
            return node.depth
        return max(self.max_depth(child) for child in node.children.values())

    def avg_depth(self, node, total_nodes):
        def sum_depths(n):
            return n.depth + sum(sum_depths(c) for c in n.children.values())
        return sum_depths(node) / total_nodes if total_nodes > 0 else 0

    def branching_factor(self, node, total_nodes):
        def count_internal(n):
            if not n.children:
                return 0
            return 1 + sum(count_internal(c) for c in n.children.values())

        def count_edges(n):
            return len(n.children) + sum(count_edges(c) for c in n.children.values())

        internal = count_internal(node)
        edges = count_edges(node)
        return edges / internal if internal > 0 else 0

    def reproduce_state(self, node):
        """Reproduce el estado del environment para llegar a un nodo"""
        self.env.reset([0])

        # Reproduce todas las acciones desde raíz hasta node
        action_sequence = node.get_action_sequence()
        for action_idx in action_sequence:
            action = torch.tensor([action_idx], device=self.env.device)
            self.env.step(action)

    def run(self):
        print("="*90)
        print(f"MCTS CORRECTO (reproduce acciones desde raíz)")
        print(f"Iteraciones: {self.n_iterations}, c={self.c}")
        print("="*90 + "\n")

        for iteration in range(self.n_iterations):
            # Reset a raíz
            self.env.reset([0])
            node = self.root
            done = False
            path = [(self.root, None, 0.0)]  # La raíz siempre se actualiza
            selection_depth = 0

            # ─── SELECTION ──────────────────────────────────────────────
            # Solo desciende si: está FULLY_EXPANDED (sin untried_actions) Y tiene hijos
            while node.is_fully_expanded() and node.children and not done:
                best_child = node.best_child(c=self.c)
                if best_child is None:
                    break

                # El env ya está en el estado correcto tras env.reset() al inicio
                # y steps secuenciales de iteraciones previas del while

                action = torch.tensor([best_child.action], device=self.env.device)
                reward, done_batch = self.env.step(action)
                done = done_batch[0].item()

                # En path, registra el nodo que fue visitado (best_child)
                path.append((best_child, action, reward[0].item()))
                node = best_child
                selection_depth = node.depth

            # ─── EXPANSION ──────────────────────────────────────────────
            if node.untried_actions is None:
                node.init_untried_actions(self.env.n_actions)

            if node.untried_actions and not done:
                action_idx = node.untried_actions.pop()
                action = torch.tensor([action_idx], device=self.env.device)
                reward, done_batch = self.env.step(action)
                done = done_batch[0].item()

                child = MCTSNodeCorrect(parent=node, action=action_idx,
                                       depth=node.depth + 1)
                # Si el nodo es terminal (done=True), no inicializar untried_actions
                # Esto hace que is_fully_expanded() = True (consistente con terminal)
                if not done:
                    child.init_untried_actions(self.env.n_actions)
                else:
                    child.untried_actions = []  # Terminal: no tiene acciones posibles
                node.children[action_idx] = child
                path.append((child, action_idx, reward[0].item()))
                node = child

            # ─── SIMULATION ─────────────────────────────────────────────
            rollout_reward = 0.0
            if not done:
                for _ in range(self.env.CC - self.env.cursor_pos[0].item()):
                    action = torch.randint(0, self.env.n_actions, (1,), device=self.env.device)
                    reward, done_batch = self.env.step(action)
                    rollout_reward += reward[0].item()
                    if done_batch[0].item():
                        break

            # ─── BACKPROP ────────────────────────────────────────────────
            path_reward = sum(r for _, _, r in path)
            total_reward = path_reward + rollout_reward
            for node_in_path, _, _ in path:
                node_in_path.update(total_reward)

            # ─── METRICS ─────────────────────────────────────────────────
            self.iteration_rewards.append(total_reward)
            best = max(self.iteration_rewards)
            self.best_rewards.append(best)

            if total_reward > self.best_reward_found:
                self.best_reward_found = total_reward
                # Guarda la mejor política encontrada
                state = self.env.get_single_state(0)
                self.best_policy = state['suma_grid']

            total_nodes = self.count_nodes(self.root)
            max_d = self.max_depth(self.root)
            avg_d = self.avg_depth(self.root, total_nodes)
            bf = self.branching_factor(self.root, total_nodes)

            self.tree_stats['total_nodes'].append(total_nodes)
            self.tree_stats['max_depth'].append(max_d)
            self.tree_stats['avg_depth'].append(avg_d)
            self.tree_stats['root_children'].append(len(self.root.children))
            self.tree_stats['branching_factor'].append(bf)

            # ─── TENSORBOARD LOGGING ────────────────────────────────────
            self.writer.add_scalar('Reward/iteration', total_reward, iteration + 1)
            self.writer.add_scalar('Reward/best_so_far', best, iteration + 1)
            self.writer.add_scalar('Reward/avg_100', np.mean(self.iteration_rewards[-100:]), iteration + 1)

            self.writer.add_scalar('Tree/total_nodes', total_nodes, iteration + 1)
            self.writer.add_scalar('Tree/max_depth', max_d, iteration + 1)
            self.writer.add_scalar('Tree/avg_depth', avg_d, iteration + 1)
            self.writer.add_scalar('Tree/root_children', len(self.root.children), iteration + 1)
            self.writer.add_scalar('Tree/branching_factor', bf, iteration + 1)
            self.writer.add_scalar('Tree/selection_depth', selection_depth, iteration + 1)

            if (iteration + 1) % max(1, self.n_iterations // 20) == 0:
                avg_recent = np.mean(self.iteration_rewards[-100:])
                print(f"  [{iteration+1:5d}] reward={total_reward:+.4f} best={best:+.4f} "
                      f"avg100={avg_recent:+.4f} | nodes={total_nodes:4d} depth={max_d} "
                      f"bf={bf:.2f}")

        print("\n✅ MCTS completado\n")
        self.writer.close()

    def get_best_policy_grid(self):
        if self.best_policy is None:
            return None
        grid_flat = [self.env.possible_actions.index(a) if a in self.env.possible_actions
                     else -1 for a in self.best_policy]
        grid_2d = np.array(grid_flat).reshape(self.env.height, self.env.grid_size)
        return grid_2d

    def visualize_policy(self, save_path=None):
        grid_2d = self.get_best_policy_grid()
        if grid_2d is None:
            print("❌ No hay política para visualizar")
            return

        print("\n" + "="*80)
        print("MEJOR POLÍTICA ENCONTRADA")
        print("="*80)
        print(f"Reward: {self.best_reward_found:+.4f}\n")

        # Mostrar productos parciales disponibles
        print("Productos parciales disponibles:")
        print("-" * 80)
        for i, action in enumerate(self.env.possible_actions):
            print(f"  {i:2d}: {action}")
        print("-" * 80 + "\n")

        # Tabla de política (por índice de acción)
        print("Política Mejor (índice de acción por celda):\n")
        print("      ", end="")
        for col in range(self.env.grid_size):
            print(f" Col{col} ", end="")
        print()
        print("     " + "-" * (8 * self.env.grid_size))

        for row in range(self.env.height):
            print(f"Row{row}  ", end="")
            for col in range(self.env.grid_size):
                action_idx = grid_2d[row, col]
                if action_idx < 0:
                    print(f"  -  ", end="")
                else:
                    print(f" {action_idx:2d} ", end="")
            print()

        print("\nPolítica Mejor (Productos parciales por celda):\n")
        col_width = 15
        print("      ", end="")
        for col in range(self.env.grid_size):
            print(f"Col{col:2d}{' '*(col_width-5)}", end="")
        print()
        print("     " + "-" * (col_width * self.env.grid_size))

        for row in range(self.env.height):
            print(f"Row{row}  ", end="")
            for col in range(self.env.grid_size):
                action_idx = grid_2d[row, col]
                if action_idx < 0:
                    action_name = "-"
                else:
                    action_name = self.env.possible_actions[action_idx]
                print(f"{action_name:<{col_width}}", end="")
            print()

        print("\n" + "="*80)

        if save_path:
            self._plot_and_save(save_path)

    def _plot_and_save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'MCTS Correcto - Bits={self.env.Bits}, c={self.c}',
                     fontsize=14, fontweight='bold')

        # Plot 1: Rewards
        axes[0, 0].plot(self.iteration_rewards, alpha=0.5, label='Iteration Reward')
        axes[0, 0].plot(self.best_rewards, label='Best So Far', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward over Iterations')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Tree Growth
        axes[0, 1].plot(self.tree_stats['total_nodes'], label='Total Nodes', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Node Count')
        axes[0, 1].set_title('Tree Growth')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Depth
        axes[0, 2].plot(self.tree_stats['max_depth'], label='Max Depth', linewidth=2)
        axes[0, 2].plot(self.tree_stats['avg_depth'], label='Avg Depth', alpha=0.7)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Depth')
        axes[0, 2].set_title('Tree Depth')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Plot 4: Branching Factor
        axes[1, 0].plot(self.tree_stats['branching_factor'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Branching Factor')
        axes[1, 0].set_title('Average Branching Factor')
        axes[1, 0].grid(True)

        # Plot 5: Root Children
        axes[1, 1].plot(self.tree_stats['root_children'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Root Children')
        axes[1, 1].set_title('Root Direct Children')
        axes[1, 1].grid(True)

        # Plot 6: Moving Average
        window = 100
        if len(self.iteration_rewards) >= window:
            moving_avg = np.convolve(self.iteration_rewards, np.ones(window)/window, mode='valid')
            axes[1, 2].plot(moving_avg, label=f'{window}-iter moving avg', linewidth=2)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].set_title('Moving Average Reward')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/mcts_analysis.png', dpi=150)
        print(f"✅ Gráficos guardados en {save_dir}/mcts_analysis.png")

        # Guardar estadísticas en JSON
        stats_json = {
            'best_reward': float(self.best_reward_found),
            'final_nodes': self.count_nodes(self.root),
            'final_depth': self.max_depth(self.root),
            'iterations': self.n_iterations,
            'c_exploration': self.c,
            'bits': self.env.Bits,
            'height': self.env.height,
        }
        with open(f'{save_dir}/stats.json', 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"✅ Estadísticas guardadas en {save_dir}/stats.json")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MCTS Correcto para Multiplicadores Binarios')
    parser.add_argument('--bits', type=int, default=2, help='Bits del multiplicador')
    parser.add_argument('--height', type=int, default=2, help='Altura de tabla')
    parser.add_argument('--iterations', type=int, default=2000, help='Iteraciones MCTS')
    parser.add_argument('--c', type=float, default=2.0, help='Coeficiente exploración UCT')
    parser.add_argument('--output', type=str, default='./mcts_results', help='Directorio salida')
    parser.add_argument('--no-tensorboard', action='store_true', help='Sin TensorBoard')

    args = parser.parse_args()

    # Crear environment
    env = BinaryMathEnvCUDA(Bits=args.bits, height=args.height, n_envs=1, device='cuda')

    # Ejecutar MCTS
    mcts = MCTSCorrectDesign(env, n_iterations=args.iterations, c=args.c, log_dir='./runs')
    mcts.run()

    # Visualizar política
    mcts.visualize_policy(args.output)

    print(f"\n{'='*90}")
    print(f"RESUMEN FINAL")
    print(f"{'='*90}")
    print(f"Total nodes: {mcts.count_nodes(mcts.root)}")
    print(f"Max depth: {mcts.max_depth(mcts.root)}")
    print(f"Best reward: {mcts.best_reward_found:+.4f}")
    print(f"{'='*90}")
