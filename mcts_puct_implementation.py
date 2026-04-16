#!/usr/bin/env python3
"""
PUCT (Polynomial Upper Confidence Bounds applied to Trees)

PUCT es una evolución de UCT para espacios de búsqueda muy grandes.

Diferencia clave:
  - UCT: exploration ∝ sqrt(ln(parent_visits) / child_visits)
    → Crece lentamente, favorece nodes poco visitados indefinidamente

  - PUCT: exploration ∝ sqrt(parent_visits) / child_visits
    → Polinomial, reduce exploración más drásticamente
    → Mejor para espacios grandes donde hay muchas "trampas"

Variantes implementadas:
  1. PUCT_POLY2: Polinomial grado 2, suave
     exploration = C * sqrt(parent_visits) / (1 + child_visits)

  2. PUCT_EXP: Exponencial (AlphaGo style)
     exploration = C * exp(-child_visits / K)
     → Reduce drásticamente exploración conforme se visita

Uso:
  python3 mcts_puct_implementation.py --variant poly2 --bits 8 --iterations 10000
  python3 mcts_puct_implementation.py --variant exp --bits 8 --iterations 10000
  python3 mcts_puct_implementation.py --variant both --bits 8 --iterations 10000
"""

import time
import random
import numpy as np
import torch
from math import sqrt, log, exp
from collections import defaultdict
import argparse
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from Environment import BinaryMathEnv
from mcts_logger import MCTSLogger


# =============================================================================
# Nodo PUCT
# =============================================================================

class PUCTNode:
    """Nodo de árbol PUCT."""

    def __init__(self, env, parent=None, action=None, puct_variant='poly2'):
        self.state = env.get_state()
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward_sum = 0.0
        self.reward_max = -float('inf')
        self.reward_min = float('inf')
        self.untried_actions = list(range(len(env.possible_actions)))
        random.shuffle(self.untried_actions)
        self.puct_variant = puct_variant

    def puct_value(self, C=1.41, K=1.0):
        """
        Calcula valor PUCT para selection.

        Args:
            C: parámetro de exploración
            K: parámetro de decay (para exponencial)

        Returns:
            valor de selection
        """
        if self.visits == 0:
            return float('inf')

        exploit = self.reward_sum / self.visits

        if self.puct_variant == 'poly2':
            # Polinomial grado 2
            if self.parent is None or self.parent.visits == 0:
                explore = 0.0
            else:
                explore = C * sqrt(self.parent.visits) / (1.0 + self.visits)

        elif self.puct_variant == 'exp':
            # Exponencial
            explore = C * exp(-self.visits / K)

        else:
            raise ValueError(f"Unknown variant: {self.puct_variant}")

        return exploit + explore

    def best_child(self, C=1.41, K=1.0):
        """Selecciona mejor hijo por PUCT."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.puct_value(C, K))

    def expand(self, env):
        """Expande una acción no explorada."""
        if not self.untried_actions:
            return None, 0.0, False

        action = self.untried_actions.pop()
        env.set_state(self.state)
        _, reward, terminated, truncated, _ = env.step(action)

        child = PUCTNode(env, parent=self, action=action,
                        puct_variant=self.puct_variant)
        self.children.append(child)
        return child, reward, terminated or truncated

    def update(self, reward):
        """Actualiza estadísticas."""
        self.visits += 1
        self.reward_sum += reward
        self.reward_max = max(self.reward_max, reward)
        self.reward_min = min(self.reward_min, reward)

    @property
    def avg_reward(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0

    def __repr__(self):
        return (f"PUCTNode(visits={self.visits}, "
                f"avg_reward={self.avg_reward:.3f})")


# =============================================================================
# Motor PUCT
# =============================================================================

class MCTSWithPUCT:
    """
    Motor de búsqueda MCTS usando PUCT en lugar de UCT.

    Parámetros:
      - variant: 'poly2' o 'exp'
      - C: parámetro de exploración
      - K: parámetro de decay (para exponencial)
    """

    def __init__(self, env, n_rollouts=50, variant='poly2',
                 C_init=1.41, K=1.0, use_dynamic_C=False):
        """
        Args:
            variant: 'poly2' o 'exp'
            C_init: parámetro de exploración inicial
            K: parámetro de decay (para exponencial)
            use_dynamic_C: si True, C decae con el tiempo (como en v2)
        """
        self.env = env
        self.n_rollouts = n_rollouts
        self.variant = variant
        self.C_init = C_init
        self.K = K
        self.use_dynamic_C = use_dynamic_C

        self.root = PUCTNode(env, puct_variant=variant)
        self.iteration = 0
        self.total_iterations = 1

        # Logging
        self.logger = MCTSLogger(strategy_name=f'puct_{variant}', log_interval=100)
        self.best_reward_logged = -float('inf')
        self.rewards_history = []
        self.best_rewards_history = []
        self.tree_sizes = []

    def search(self, iterations=10000, verbose=True):
        """Búsqueda principal."""
        self.total_iterations = iterations
        print(f"\n{'='*80}")
        print(f"PUCT ({self.variant}) - Iniciando búsqueda ({iterations} iteraciones)")
        print(f"C_init={self.C_init}, K={self.K}, use_dynamic_C={self.use_dynamic_C}")
        print(f"{'='*80}\n")

        for it in range(iterations):
            self.iteration += 1

            # Determinar C actual
            if self.use_dynamic_C:
                C_current = self.C_init * (1.0 - it / iterations) ** 2
            else:
                C_current = self.C_init

            # Selection + Expansion
            node, step_reward, done = self._select_and_expand(C_current)

            # Simulation (CPU serial rollout)
            if not done:
                rollout_reward = self._simulate(node)
            else:
                rollout_reward = step_reward

            total_reward = step_reward + rollout_reward

            # Backpropagation
            self._backpropagate(node, total_reward)

            # Logging
            if it % 100 == 0:
                best_reward = self._get_best_reward()
                tree_size = self._count_nodes()

                self.rewards_history.append(total_reward)
                self.best_rewards_history.append(best_reward)
                self.tree_sizes.append(tree_size)

                # Log con MCTSLogger
                self.logger.log_standard(it, total_reward, best_reward, tree_size)
                self.logger.log_strategy_scalars(it, C_current=C_current, root_visits=self.root.visits)

                # Topology cada 500 iters
                if it % 500 == 0:
                    self.logger.log_topology(it, self.root.state['suma_grid'],
                                            self.root.state.get('height', 8),
                                            self.root.state.get('Bits', 8))

                # Policy cuando mejora
                if best_reward > self.best_reward_logged:
                    self.best_reward_logged = best_reward
                    policy = self.get_policy()
                    self.logger.log_policy_text(it, policy)

                if verbose:
                    print(f"[{self.variant:>4}] It {it:6d}/{iterations}: "
                          f"r={total_reward:7.3f}, best={best_reward:7.3f}, "
                          f"tree={tree_size:6d}, C={C_current:.3f}")

        self.logger.close()

        best = self._get_best_reward()
        print(f"\n✅ PUCT ({self.variant}) completado")
        print(f"   Mejor reward alcanzado: {best:.4f}")
        print(f"   Tamaño final del árbol: {self._count_nodes()}")

        return best

    def _select_and_expand(self, C):
        """Selection y expansion."""
        node = self.root

        while node.children or node.untried_actions:
            if node.untried_actions:
                return node.expand(self.env)

            node = node.best_child(C, self.K)
            if node is None:
                break

        return node, 0.0, False

    def _backpropagate(self, node, reward):
        """Backpropagation."""
        while node is not None:
            node.update(reward)
            node = node.parent

    def _simulate(self, node):
        """
        Simulación: rollout aleatorio desde el nodo.
        Usa CPU serial rollouts (como mcts_simple_benchmark.py).

        Returns:
            reward promedio de los rollouts
        """
        rewards = []
        for _ in range(self.n_rollouts):
            self.env.set_state(node.state)
            done = False
            for step in range(self.env.CC):
                action = random.randint(0, len(self.env.possible_actions) - 1)
                _, reward, done, truncated, _ = self.env.step(action)
                if done or truncated:
                    rewards.append(reward)
                    break
            else:
                # No completó la tabla
                rewards.append(-10.0)

        return float(np.mean(rewards)) if rewards else -10.0

    def _get_best_reward(self):
        """Mejor reward visto."""
        best = [-float('inf')]
        def dfs(node):
            if node.avg_reward > best[0]:
                best[0] = node.avg_reward
            for child in node.children:
                dfs(child)
        dfs(self.root)
        return best[0]

    def _count_nodes(self):
        """Cuenta nodos totales."""
        def dfs(node):
            return 1 + sum(dfs(c) for c in node.children)
        return dfs(self.root)

    def get_policy(self):
        """
        Extrae la política greedy del árbol (max visits por nivel).

        Returns:
            lista de dicts {step, action_idx, action_name, visits, reward_mean}
        """
        policy = []
        node = self.root
        step = 0

        while node.children:
            # Buscar hijo con máximo número de visitas
            best_child = max(node.children, key=lambda c: c.visits)
            if best_child is None:
                break

            policy.append({
                'step': step,
                'action_idx': best_child.action,
                'action_name': self.env.possible_actions[best_child.action],
                'visits': best_child.visits,
                'reward_mean': best_child.avg_reward,
            })

            node = best_child
            step += 1

        return policy

    def get_convergence_speed(self, target_reward=-0.5):
        """Iteración en que se alcanzó target."""
        for i, r in enumerate(self.best_rewards_history):
            if r >= target_reward:
                return (i + 1) * 100  # Multiplicar por 100 porque logeamos cada 100
        return None


# =============================================================================
# Comparación PUCT Poly2 vs Exp
# =============================================================================

def compare_puct_variants(bits=8, iterations=10000, n_rollouts=50):
    """Compara ambas variantes de PUCT."""

    print("\n" + "="*80)
    print("COMPARACIÓN DE VARIANTES PUCT")
    print("="*80)

    env = BinaryMathEnv(Bits=bits, Proof=4, height=8)
    env.reset()

    results = {}

    # PUCT Poly2
    print("\n[1/2] PUCT Polinomial (grado 2)...")
    mcts_poly2 = MCTSWithPUCT(
        env=env.clone(),
        n_rollouts=n_rollouts,
        variant='poly2',
        C_init=1.41,
        use_dynamic_C=True
    )
    start = time.time()
    best_poly2 = mcts_poly2.search(iterations, verbose=False)
    time_poly2 = time.time() - start
    results['puct_poly2'] = {
        'best_reward': best_poly2,
        'time': time_poly2,
        'convergence_speed': mcts_poly2.get_convergence_speed(-0.5)
    }

    # PUCT Exp
    print("\n[2/2] PUCT Exponencial (AlphaGo style)...")
    mcts_exp = MCTSWithPUCT(
        env=env.clone(),
        n_rollouts=n_rollouts,
        variant='exp',
        C_init=1.0,
        K=50.0,  # Parámetro de decay
        use_dynamic_C=False
    )
    start = time.time()
    best_exp = mcts_exp.search(iterations, verbose=False)
    time_exp = time.time() - start
    results['puct_exp'] = {
        'best_reward': best_exp,
        'time': time_exp,
        'convergence_speed': mcts_exp.get_convergence_speed(-0.5)
    }

    # Resultados
    print("\n" + "="*80)
    print("RESULTADOS COMPARATIVOS")
    print("="*80)

    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Mejor Reward: {data['best_reward']:.4f}")
        print(f"  Tiempo Total: {data['time']:.1f}s")
        print(f"  Iteraciones/seg: {iterations/data['time']:.1f}")
        conv = data['convergence_speed']
        if conv:
            print(f"  Convergencia a -0.5: {conv} iteraciones")
        else:
            print(f"  No converged to -0.5")

    winner = max(results.items(), key=lambda x: x[1]['best_reward'])
    print(f"\n🏆 Ganador: {winner[0]} (reward={winner[1]['best_reward']:.4f})")

    if TENSORBOARD_AVAILABLE:
        print(f"\n📊 Logs guardados en ./logs_mcts/")
        print(f"   tensorboard --logdir ./logs_mcts/")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['poly2', 'exp', 'both'],
                       default='both', help='Variante PUCT a usar')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--rollouts', type=int, default=512)
    args = parser.parse_args()

    env = BinaryMathEnv(Bits=args.bits, Proof=4, height=8)
    env.reset()

    if args.variant == 'both':
        compare_puct_variants(args.bits, args.iterations, args.rollouts)
    else:
        mcts = MCTSWithPUCT(
            env=env,
            n_rollouts=args.rollouts,
            variant=args.variant,
            C_init=1.41 if args.variant == 'poly2' else 1.0,
            K=1.0 if args.variant == 'poly2' else 50.0,
            use_dynamic_C=True
        )
        mcts.search(args.iterations)
