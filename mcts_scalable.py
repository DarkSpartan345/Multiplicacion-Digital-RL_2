#!/usr/bin/env python3
"""
MCTS Escalable - Progressive Widening + GPU Parallel Rollouts + State Caching

Mejoras sobre mcts_correct_design.py para escalar a Bits >= 4:

1. Progressive Widening: limita hijos por nodo a C_pw * visits^alpha,
   permitiendo alcanzar profundidad maxima sin explorar exponencialmente.
   Con n_actions=66 (Bits=4), un nodo no necesita expandir las 66 acciones
   antes de descender -- solo ceil(C_pw * visits^0.5) hijos son suficientes.

2. GPU Parallel Rollouts: N rollouts simultaneos en CUDA para la fase de
   simulacion, promediando rewards para mejor estimacion de valor.

3. State Caching: estado GPU (grid, cursor, carry) almacenado directamente
   en cada nodo del arbol. Elimina reproduce_state() O(D). La seleccion
   sigue punteros sin tocar el environment (O(1) por nivel).

Flujo por iteracion:
  SELECCION:  sigue punteros del arbol, sin env interaction
  EXPANSION:  restaura estado cacheado -> env.step una vez -> cachea hijo
  SIMULACION: restaura estado hijo en N slots GPU -> rollouts paralelos
  BACKPROP:   actualiza reward promedio en todo el camino
"""

import torch
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '/home/servergmun/MCTS_CUCA_CACA')
from Environment import BinaryMathEnvCUDA


class CachedState:
    """Snapshot del estado GPU de un slot del environment.

    Almacena tensores clonados en GPU para restauracion O(1).
    Memoria por nodo (Bits=4): ~1.1KB (grid 64B + carry 1024B + scalars).
    Para Bits=8+: carry crece a 256KB/nodo -- considerar almacenamiento en CPU.
    """
    __slots__ = ['grid', 'cursor', 'reward', 'is_done', 'carry']

    def __init__(self, env, slot=0):
        self.grid = env.suma_grid[slot].clone()
        self.cursor = int(env.cursor_pos[slot].item())
        self.reward = float(env.rewards[slot].item())
        self.is_done = bool(env.done[slot].item())
        self.carry = env.carry_in[slot].clone() if env.incremental else None


class MCTSNodePW:
    """Nodo MCTS con progressive widening y estado cacheado en GPU.

    Progressive widening: max_children = ceil(C_pw * visits^alpha).
    Con C_pw=2.0, alpha=0.5:
      visits=1  -> max 2 hijos
      visits=9  -> max 6 hijos
      visits=100 -> max 20 hijos
      visits=1089 -> max 66 hijos (todas las acciones para Bits=4)
    """
    __slots__ = ['parent', 'action', 'depth', 'C_pw', 'alpha',
                 'children', 'visits', 'reward_sum', 'untried_actions',
                 'cached_state']

    def __init__(self, parent=None, action=None, depth=0, C_pw=2.0, alpha=0.5):
        self.parent = parent
        self.action = action
        self.depth = depth
        self.C_pw = C_pw
        self.alpha = alpha
        self.children = {}
        self.visits = 0
        self.reward_sum = 0.0
        self.untried_actions = None
        self.cached_state = None

    def init_untried_actions(self, n_actions):
        if self.untried_actions is None:
            self.untried_actions = list(range(n_actions))
            random.shuffle(self.untried_actions)

    def max_children_allowed(self):
        if self.visits <= 0:
            return 1
        return max(1, int(math.ceil(self.C_pw * (self.visits ** self.alpha))))

    def should_expand(self):
        """True si el nodo debe expandir un hijo nuevo (progressive widening).

        Retorna False si:
        - No hay acciones sin probar (untried_actions vacio/None)
        - Ya tiene suficientes hijos para su nivel de visitas
        """
        if not self.untried_actions:
            return False
        return len(self.children) < self.max_children_allowed()

    def uct_value(self, c=1.41):
        if self.visits == 0:
            return float('inf')
        exploit = self.reward_sum / self.visits
        if self.parent is None or self.parent.visits == 0:
            return exploit
        explore = c * math.sqrt(math.log(self.parent.visits) / (1 + self.visits))
        return exploit + explore

    def best_child(self, c=1.41):
        if not self.children:
            return None
        return max(self.children.values(), key=lambda ch: ch.uct_value(c))

    def update(self, reward):
        self.visits += 1
        self.reward_sum += reward

    def avg_reward(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0


class MCTSScalable:
    """MCTS con Progressive Widening, rollouts GPU paralelos, y state caching.

    Args:
        env:          BinaryMathEnvCUDA con n_envs >= n_rollouts.
        n_iterations: iteraciones MCTS.
        c:            coeficiente exploracion UCT.
        n_rollouts:   rollouts paralelos en GPU para simulacion.
        C_pw:         coeficiente progressive widening.
        alpha:        exponente progressive widening (0.5 = sqrt).
        log_dir:      directorio para TensorBoard.
    """

    def __init__(self, env, n_iterations=10000, c=2.0, n_rollouts=32,
                 C_pw=2.0, alpha=0.5, log_dir='./runs'):
        self.env = env
        self.n_iterations = n_iterations
        self.c = c
        self.n_rollouts = n_rollouts
        self.C_pw = C_pw
        self.alpha = alpha

        # Tensor pre-computado para restaurar estado a todos los slots
        self._rollout_idx = torch.arange(n_rollouts, device=env.device,
                                         dtype=torch.long)
        self._slot0_idx = torch.tensor([0], device=env.device, dtype=torch.long)

        # Root
        self.root = MCTSNodePW(C_pw=C_pw, alpha=alpha)
        self.root.init_untried_actions(env.n_actions)
        self.env.reset(list(range(n_rollouts)))
        self.root.cached_state = CachedState(env, slot=0)

        # Contadores incrementales (evitan recorrer arbol cada iteracion)
        self._node_count = 1
        self._max_depth_seen = 0

        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = (f"mcts_pw_bits{env.Bits}_c{c:.2f}_"
                    f"r{n_rollouts}_pw{C_pw}_{timestamp}")
        self.writer = SummaryWriter(f'{log_dir}/{run_name}')
        print(f"\nTensorBoard: tensorboard --logdir {log_dir}")
        print(f"Run: {run_name}")
        print(f"Progressive Widening: C_pw={C_pw}, alpha={alpha}")
        print(f"Parallel rollouts: {n_rollouts}")
        print(f"Search space: {env.n_actions}^{env.CC} "
              f"= {env.n_actions**min(env.CC, 20):.2e}{'...' if env.CC > 20 else ''}\n")

        # Metrics
        self.iteration_rewards = []
        self.best_rewards = []
        self.best_policy = None
        self.best_reward_found = -float('inf')
        self.elapsed_time = 0.0

    # ── State management ────────────────────────────────────────────────

    def _restore_state_single(self, cached_state):
        """Restaura un CachedState en slot 0 del env."""
        self.env.suma_grid[0] = cached_state.grid
        self.env.cursor_pos[0] = cached_state.cursor
        self.env.done[0] = cached_state.is_done
        self.env.rewards[0] = cached_state.reward
        if self.env.incremental and cached_state.carry is not None:
            self.env.carry_in[0] = cached_state.carry

    def _restore_state_all(self, cached_state):
        """Restaura un CachedState en TODOS los slots de rollout."""
        idx = self._rollout_idx
        self.env.suma_grid[idx] = cached_state.grid.unsqueeze(0)
        self.env.cursor_pos[idx] = cached_state.cursor
        self.env.done[idx] = cached_state.is_done
        self.env.rewards[idx] = cached_state.reward
        if self.env.incremental and cached_state.carry is not None:
            self.env.carry_in[idx] = cached_state.carry.unsqueeze(0)

    def _step_slot0(self, action_idx):
        """Ejecuta un paso en slot 0. Otros slots deben estar done."""
        actions = torch.zeros(self.env.n_envs, dtype=torch.long,
                              device=self.env.device)
        actions[0] = action_idx
        reward, done = self.env.step(actions)
        return reward[0].item(), bool(done[0].item())

    def _parallel_rollout(self, cached_state):
        """Ejecuta n_rollouts rollouts random paralelos desde cached_state.

        Cada slot GPU recibe el mismo estado inicial y ejecuta acciones
        aleatorias independientes hasta completar el episodio.

        Returns:
            mean_reward:  promedio de rewards finales (para backprop)
            best_reward:  mejor reward individual (para tracking)
            best_policy:  grid del mejor rollout (lista de strings)
        """
        self._restore_state_all(cached_state)

        remaining = self.env.CC - cached_state.cursor
        for _ in range(remaining):
            active = ~self.env.done[:self.n_rollouts]
            if not active.any():
                break
            actions = torch.randint(0, self.env.n_actions,
                                    (self.env.n_envs,),
                                    device=self.env.device)
            self.env.step(actions)

        rewards = self.env.rewards[:self.n_rollouts]
        mean_reward = rewards.mean().item()
        best_idx = int(rewards.argmax().item())
        best_reward = rewards[best_idx].item()
        best_state = self.env.get_single_state(best_idx)

        return mean_reward, best_reward, best_state

    # ── Tree statistics (incremental + periodic full scan) ──────────────

    def _branching_factor(self, node):
        def _count(n):
            if not n.children:
                return 0, 0
            internal, edges = 1, len(n.children)
            for c in n.children.values():
                ci, ce = _count(c)
                internal += ci
                edges += ce
            return internal, edges
        internal, edges = _count(node)
        return edges / internal if internal > 0 else 0

    def _avg_depth(self, node):
        total_depth = [0]
        count = [0]
        def _walk(n):
            total_depth[0] += n.depth
            count[0] += 1
            for c in n.children.values():
                _walk(c)
        _walk(node)
        return total_depth[0] / count[0] if count[0] > 0 else 0

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        env = self.env
        print("=" * 90)
        print(f"MCTS ESCALABLE (PW + GPU Rollouts + State Cache)")
        print(f"Bits={env.Bits}, height={env.height}, CC={env.CC}, "
              f"n_actions={env.n_actions}")
        print(f"iterations={self.n_iterations}, c={self.c}, "
              f"rollouts={self.n_rollouts}")
        print("=" * 90 + "\n")

        log_interval = max(1, self.n_iterations // 20)
        stats_interval = max(1, self.n_iterations // 100)

        # Variables para stats periodicos
        last_bf = 0.0
        last_avg_d = 0.0

        t_start = time.time()

        for iteration in range(self.n_iterations):
            node = self.root
            path = [self.root]

            # ── SELECTION (O(1) por nivel, sin env interaction) ─────
            while node.children and not node.cached_state.is_done:
                if node.should_expand():
                    break
                best = node.best_child(c=self.c)
                if best is None:
                    break
                path.append(best)
                node = best

            selection_depth = node.depth

            # ── EXPANSION ───────────────────────────────────────────
            if (not node.cached_state.is_done
                    and node.should_expand()
                    and node.untried_actions):
                # Solo slot 0 activo para expansion
                self.env.done[:] = True
                self.env.done[0] = False
                self._restore_state_single(node.cached_state)

                action_idx = node.untried_actions.pop()
                _, done_val = self._step_slot0(action_idx)

                child = MCTSNodePW(
                    parent=node, action=action_idx,
                    depth=node.depth + 1,
                    C_pw=self.C_pw, alpha=self.alpha)
                child.cached_state = CachedState(self.env, slot=0)

                if not done_val:
                    child.init_untried_actions(env.n_actions)
                else:
                    child.untried_actions = []

                node.children[action_idx] = child
                path.append(child)
                node = child

                # Contadores incrementales
                self._node_count += 1
                if child.depth > self._max_depth_seen:
                    self._max_depth_seen = child.depth

            # ── SIMULATION (rollouts paralelos en GPU) ──────────────
            if node.cached_state.is_done:
                total_reward = node.cached_state.reward
                rollout_best_reward = node.cached_state.reward
                rollout_best_policy = None
            else:
                total_reward, rollout_best_reward, rollout_best_state = \
                    self._parallel_rollout(node.cached_state)
                rollout_best_policy = rollout_best_state['suma_grid']

            # ── BACKPROPAGATION ─────────────────────────────────────
            for n in path:
                n.update(total_reward)

            # ── TRACK BEST ──────────────────────────────────────────
            self.iteration_rewards.append(total_reward)
            best_so_far = max(self.iteration_rewards)
            self.best_rewards.append(best_so_far)

            if rollout_best_reward > self.best_reward_found:
                self.best_reward_found = rollout_best_reward
                if rollout_best_policy is not None:
                    self.best_policy = rollout_best_policy

            # ── TENSORBOARD ─────────────────────────────────────────
            i = iteration + 1
            self.writer.add_scalar('Reward/iteration', total_reward, i)
            self.writer.add_scalar('Reward/best_so_far', best_so_far, i)
            self.writer.add_scalar('Reward/avg_100',
                                   np.mean(self.iteration_rewards[-100:]), i)
            self.writer.add_scalar('Tree/total_nodes', self._node_count, i)
            self.writer.add_scalar('Tree/max_depth',
                                   self._max_depth_seen, i)
            self.writer.add_scalar('Tree/root_children',
                                   len(self.root.children), i)
            self.writer.add_scalar('Tree/selection_depth',
                                   selection_depth, i)

            # Stats costosos solo periodicamente
            if i % stats_interval == 0:
                last_bf = self._branching_factor(self.root)
                last_avg_d = self._avg_depth(self.root)
                self.writer.add_scalar('Tree/branching_factor', last_bf, i)
                self.writer.add_scalar('Tree/avg_depth', last_avg_d, i)

            # ── LOG ─────────────────────────────────────────────────
            if i % log_interval == 0:
                avg_recent = np.mean(self.iteration_rewards[-100:])
                print(
                    f"  [{i:6d}] reward={total_reward:+.4f} "
                    f"best={best_so_far:+.4f} "
                    f"avg100={avg_recent:+.4f} | "
                    f"nodes={self._node_count:6d} "
                    f"depth={self._max_depth_seen} "
                    f"bf={last_bf:.2f} "
                    f"sel_d={selection_depth}")

        self.elapsed_time = time.time() - t_start
        print(f"\n  MCTS completado en {self.elapsed_time:.1f}s\n")
        self.writer.close()

    # ── Visualization ───────────────────────────────────────────────────

    def visualize_policy(self, save_path=None):
        if self.best_policy is None:
            print("No hay politica para visualizar")
            return

        env = self.env
        grid_flat = self.best_policy

        print("\n" + "=" * 80)
        print("MEJOR POLITICA ENCONTRADA")
        print("=" * 80)
        print(f"Reward: {self.best_rewards[-1]:+.4f}\n")

        col_width = max(len(str(a)) for a in grid_flat) + 2
        col_width = max(col_width, 8)

        header = "      "
        for col in range(env.grid_size):
            header += f"{'Col' + str(col):<{col_width}}"
        print(header)
        print("     " + "-" * (col_width * env.grid_size))

        for row in range(env.height):
            line = f"Row{row}  "
            for col in range(env.grid_size):
                idx = row * env.grid_size + col
                val = grid_flat[idx] if idx < len(grid_flat) else "?"
                line += f"{val:<{col_width}}"
            print(line)

        print("\n" + "=" * 80)

        if save_path:
            self._plot_and_save(save_path)

    def _plot_and_save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f'MCTS Escalable - Bits={self.env.Bits}, c={self.c}, '
            f'rollouts={self.n_rollouts}, C_pw={self.C_pw}, '
            f'alpha={self.alpha}',
            fontsize=13, fontweight='bold')

        # Rewards
        axes[0, 0].plot(self.iteration_rewards, alpha=0.3, label='Iteration')
        axes[0, 0].plot(self.best_rewards, label='Best', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Tree growth (incremental)
        nodes_series = [1]  # root
        for r in self.iteration_rewards[:-1]:
            nodes_series.append(nodes_series[-1])
        # Use actual final count
        axes[0, 1].plot(range(len(self.iteration_rewards)),
                        np.linspace(1, self._node_count,
                                    len(self.iteration_rewards)),
                        linewidth=2)
        axes[0, 1].set_title('Tree Growth (approx)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Nodes')
        axes[0, 1].grid(True)

        # Best reward zoom
        axes[0, 2].plot(self.best_rewards, linewidth=2, color='red')
        axes[0, 2].set_title('Best Reward Over Time')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Best Reward')
        axes[0, 2].grid(True)

        # Reward distribution
        if len(self.iteration_rewards) > 100:
            axes[1, 0].hist(self.iteration_rewards, bins=50, alpha=0.7,
                            color='green')
            axes[1, 0].set_title('Reward Distribution')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data',
                            ha='center', va='center')

        # Root children over time
        axes[1, 1].set_title('Root Children: '
                             f'{len(self.root.children)}/{self.env.n_actions}')
        axes[1, 1].bar(range(len(self.root.children)),
                       [c.visits for c in self.root.children.values()],
                       color='orange')
        axes[1, 1].set_xlabel('Child index')
        axes[1, 1].set_ylabel('Visits')
        axes[1, 1].grid(True)

        # Moving average
        window = min(100, len(self.iteration_rewards) // 4)
        if window >= 2:
            ma = np.convolve(self.iteration_rewards,
                             np.ones(window) / window, mode='valid')
            axes[1, 2].plot(ma, linewidth=2)
        axes[1, 2].set_title(f'Moving Average ({window})')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/mcts_analysis.png', dpi=150)
        print(f"Graficos guardados en {save_dir}/mcts_analysis.png")

        stats = {
            'best_reward': float(self.best_rewards[-1]),
            'final_nodes': self._node_count,
            'final_depth': self._max_depth_seen,
            'final_branching_factor': self._branching_factor(self.root),
            'iterations': self.n_iterations,
            'c_exploration': self.c,
            'n_rollouts': self.n_rollouts,
            'C_pw': self.C_pw,
            'alpha': self.alpha,
            'bits': self.env.Bits,
            'height': self.env.height,
            'CC': self.env.CC,
            'n_actions': self.env.n_actions,
            'elapsed_time_s': round(self.elapsed_time, 2),
        }
        with open(f'{save_dir}/stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Estadisticas guardadas en {save_dir}/stats.json")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='MCTS Escalable con Progressive Widening + GPU Rollouts')
    parser.add_argument('--bits', type=int, default=4,
                        help='Bits del multiplicador (default: 4)')
    parser.add_argument('--height', type=int, default=4,
                        help='Filas de la tabla (default: 4)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Iteraciones MCTS (default: 10000)')
    parser.add_argument('--c', type=float, default=2.0,
                        help='Coeficiente exploracion UCT (default: 2.0)')
    parser.add_argument('--n-rollouts', type=int, default=32,
                        help='Rollouts paralelos GPU (default: 32)')
    parser.add_argument('--C-pw', type=float, default=2.0,
                        help='Coeficiente progressive widening (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Exponente progressive widening (default: 0.5)')
    parser.add_argument('--output', type=str,
                        default='./mcts_scalable_results',
                        help='Directorio de salida')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='Directorio TensorBoard')

    args = parser.parse_args()

    print(f"\nConfiguracion:")
    print(f"  Bits={args.bits}, height={args.height}")
    print(f"  Acciones: 4*{args.bits}^2 + 2 = {4 * args.bits**2 + 2}")
    print(f"  Episodio: {args.height} * 2 * {args.bits} = "
          f"{args.height * 2 * args.bits} pasos")
    print(f"  Test cases: {(2**args.bits)**2}")
    print()

    env = BinaryMathEnvCUDA(
        Bits=args.bits,
        height=args.height,
        n_envs=args.n_rollouts,
        device='cuda',
        incremental=True,
    )

    mcts = MCTSScalable(
        env,
        n_iterations=args.iterations,
        c=args.c,
        n_rollouts=args.n_rollouts,
        C_pw=args.C_pw,
        alpha=args.alpha,
        log_dir=args.log_dir,
    )
    mcts.run()
    mcts.visualize_policy(args.output)

    print(f"\n{'=' * 90}")
    print(f"RESUMEN FINAL")
    print(f"{'=' * 90}")
    print(f"  Bits:           {args.bits}")
    print(f"  Height:         {args.height}")
    print(f"  Iteraciones:    {args.iterations}")
    print(f"  Total nodes:    {mcts._node_count}")
    print(f"  Max depth:      {mcts._max_depth_seen} / {env.CC}")
    print(f"  Best reward:    {mcts.best_rewards[-1]:+.4f}")
    print(f"  Branching:      {mcts._branching_factor(mcts.root):.2f}")
    print(f"  Root children:  {len(mcts.root.children)} / {env.n_actions}")
    print(f"  Tiempo:         {mcts.elapsed_time:.1f}s")
    print(f"{'=' * 90}")
