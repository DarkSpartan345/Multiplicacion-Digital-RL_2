#!/usr/bin/env python3
"""
MCTS + UCT v2: Mejoras Progresivas para Convergencia

Basado en mcts_uct_v1.py, esta versión añade:

1. TRANSPORTE INVERSO (Inverse/Reverse UCB):
   - En vez de usar UCB1 para descender (selection),
     usamos la MEDIA de recompensas durante selection.
   - Beneficio: desciende más rápido a caminos prometedores.

2. AJUSTE DINÁMICO DE C:
   - C no es constante, cambia según la fase de búsqueda:
     - Early stage: C alto (mucha exploración)
     - Late stage:  C bajo (mucha explotación)
   - Beneficio: al inicio no sabemos nada, exploramos; luego explotamos.

3. PODA DÉBIL (Soft Pruning):
   - Ocasionalmente, eliminamos nodos con reward muy bajo.
   - Pero los marcamos para poder revivirlos si es necesario.
   - Beneficio: memoria y velocidad sin perder soluciones.

4. RECOMPENSA AUXILIAR (Intrinsic Reward):
   - Penalizamos estados que "se ven mal" temprano.
   - Favorecemos estados que exploran muchas opciones.
   - Beneficio: feedback más temprano, evita callejones sin salida.

Flujo en cada iteración:
  1. SELECTION: Usa media de rewards (no UCT) para descender
  2. EXPANSION: Como antes
  3. SIMULATION: Como antes, pero registra más estadísticas
  4. BACKPROP: Actualiza con recompensa auxiliar si aplica
"""

import time
import random
import numpy as np
import torch
from math import sqrt, log
from collections import defaultdict

from Environment import BinaryMathEnv, BinaryMathEnvCUDA
from mcts_logger import MCTSLogger


# =============================================================================
# MCTSNodeUCTV2 - Nodo mejorado con soporte para nuevas características
# =============================================================================

class MCTSNodeUCTV2:
    """
    Nodo MCTS con soporte para:
      - Transporte inverso
      - Recompensa auxiliar
      - Estadísticas de calidad
    """

    def __init__(self, env, parent=None, action=None):
        self.state = env.get_state()
        self.parent = parent
        self.action = action
        self.children = []

        # Estadísticas base
        self.visits = 0
        self.reward_sum = 0.0
        self.reward_min = float('inf')
        self.reward_max = -float('inf')

        # Estadísticas auxiliares
        self.intrinsic_reward_sum = 0.0  # Para ajustes dinámicos
        self.quality_score = 0.5  # Score de "promisingness" [0,1]

        # Control de poda
        self.is_pruned = False

        # Acciones
        self.untried_actions = list(range(len(env.possible_actions)))
        random.shuffle(self.untried_actions)

    def uct_value(self, C=1.41):
        """UCB1 estándar (para expansion, no selection)."""
        if self.visits == 0:
            return float('inf')
        if not self.parent:
            return 0.0
        exploit = self.reward_sum / self.visits
        explore = C * sqrt(log(self.parent.visits) / self.visits)
        return exploit + explore

    def selection_value(self, method='mean'):
        """
        Valor para usar durante SELECTION (descenso del árbol).

        Methods:
          'mean': solo la media de recompensas (transporte inverso)
          'uct': UCB1 estándar
          'quality': usa quality_score como indicador
        """
        if self.visits == 0:
            return float('inf')

        if method == 'mean':
            # Transporte inverso: descender por mejores recompensas
            return self.reward_sum / self.visits
        elif method == 'quality':
            # Usar score de calidad
            return self.quality_score
        else:
            return self.uct_value()

    def best_child_for_selection(self, method='mean'):
        """Selecciona mejor hijo durante SELECTION."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.selection_value(method))

    def best_child_for_expansion(self, C=1.41):
        """Selecciona mejor hijo durante EXPANSION (usa UCT)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct_value(C))

    def expand(self, env):
        """Expande una nueva acción."""
        if not self.untried_actions:
            return None, 0.0, False

        action = self.untried_actions.pop()
        env.set_state(self.state)
        _, reward, terminated, truncated, _ = env.step(action)

        child = MCTSNodeUCTV2(env, parent=self, action=action)
        self.children.append(child)
        return child, reward, terminated or truncated

    def update(self, reward, intrinsic_bonus=0.0):
        """
        Actualiza con recompensa + bonificación auxiliar.

        Args:
            reward: recompensa principal
            intrinsic_bonus: bonificación auxiliar por exploración
        """
        self.visits += 1
        self.reward_sum += reward
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)

        # Intrinsic reward: exploración de opciones
        self.intrinsic_reward_sum += intrinsic_bonus

        # Actualizar quality_score (mezcla de rewards + exploración)
        mean_reward = self.reward_sum / self.visits
        intrinsic_mean = self.intrinsic_reward_sum / self.visits if self.visits > 0 else 0
        self.quality_score = 0.7 * (mean_reward + 10) / 20 + 0.3 * intrinsic_mean

    @property
    def avg_reward(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def __repr__(self):
        return (f"Node(v={self.visits}, r={self.avg_reward:.2f}, "
                f"q={self.quality_score:.2f}, pruned={self.is_pruned})")


# =============================================================================
# MCTSWithUCTV2 - Motor de búsqueda mejorado
# =============================================================================

class MCTSWithUCTV2:
    """
    MCTS + UCT v2 con transporte inverso, ajuste dinámico de C,
    y recompensa auxiliar.
    """

    def __init__(self, env, cuda_env, n_rollouts=256, C_init=1.41,
                 enable_inverse_transport=True,
                 enable_dynamic_C=True,
                 enable_intrinsic_reward=True,
                 verbose=True):
        """
        Args:
            enable_inverse_transport: usar media en selection
            enable_dynamic_C: C cambia según fase de búsqueda
            enable_intrinsic_reward: bonificación auxiliar por exploración
        """
        self.env = env
        self.cuda_env = cuda_env
        self.n_rollouts = n_rollouts
        self.C_init = C_init
        self.verbose = verbose

        # Features
        self.enable_inverse_transport = enable_inverse_transport
        self.enable_dynamic_C = enable_dynamic_C
        self.enable_intrinsic_reward = enable_intrinsic_reward

        self.root = MCTSNodeUCTV2(env)
        self.iteration = 0
        self.total_iterations = 0
        self.max_reward_ever = -float('inf')

        # TensorBoard Logger
        self.logger = MCTSLogger(strategy_name='uct_v2', log_interval=50)
        self.best_reward_logged = -float('inf')

        # Métricas
        self._times = defaultdict(float)
        self._rewards = []
        self._selection_depths = []

    def _get_C(self, current_iter, total_iters):
        """
        C dinámico: comienza alto (exploración) y decae (explotación).

        Formula: C(t) = C_init * (1 - t/T)^2
        """
        if not self.enable_dynamic_C:
            return self.C_init

        progress = current_iter / max(total_iters, 1)
        decay = (1 - progress) ** 2
        return self.C_init * decay

    def search(self, iterations=500, log_every=50):
        """Ejecuta búsqueda MCTS v2."""
        if self.verbose:
            print(f"\n{'='*90}")
            print(f"  MCTS + UCT v2: Mejoras Progresivas")
            print(f"{'='*90}")
            print(f"  Transporte inverso:  {self.enable_inverse_transport}")
            print(f"  C dinámico:          {self.enable_dynamic_C}")
            print(f"  Recompensa auxiliar: {self.enable_intrinsic_reward}")
            print(f"  Parámetro C inicial: {self.C_init}")
            print()

        self.total_iterations = iterations
        t_start = time.perf_counter()

        for it in range(iterations):
            self.iteration = it

            # Obtener C actual
            C = self._get_C(it, iterations)

            # ── 1. SELECTION ──────────────────────────────────────────────────
            t1 = time.perf_counter()
            node, depth = self._select_and_expand(C)
            self._times['select'] += time.perf_counter() - t1

            if node is None:
                continue

            self._selection_depths.append(depth)

            # ── 2. SIMULATION ──────────────────────────────────────────────────
            t2 = time.perf_counter()
            reward_mean = self._simulate(node)
            self._times['simulate'] += time.perf_counter() - t2

            self.max_reward_ever = max(self.max_reward_ever, reward_mean)
            self._rewards.append(reward_mean)

            # ── 3. INTRINSIC REWARD ────────────────────────────────────────────
            # Bonificación por exploración temprana
            intrinsic_bonus = 0.0
            if self.enable_intrinsic_reward:
                intrinsic_bonus = self._compute_intrinsic_bonus(node, it, iterations)

            # ── 4. BACKPROPAGATION ─────────────────────────────────────────────
            t3 = time.perf_counter()
            self._backpropagate(node, reward_mean, intrinsic_bonus)
            self._times['backprop'] += time.perf_counter() - t3

            # ── LOG ────────────────────────────────────────────────────────────
            if self.verbose and ((it + 1) % log_every == 0):
                self._print_summary(it + 1, iterations, t_start, C)

                # Log con TensorBoard
                tree_size = self._count_nodes()
                avg_reward = np.mean(self._rewards[-100:]) if self._rewards else 0
                avg_depth = np.mean(self._selection_depths[-100:]) if self._selection_depths else 0

                self.logger.log_standard(it + 1, avg_reward, self.max_reward_ever, tree_size, avg_depth)
                self.logger.log_strategy_scalars(it + 1, dynamic_C=C)

                # Topology cada 200 iters
                if (it + 1) % 200 == 0:
                    self.logger.log_topology(it + 1, self.root.state['suma_grid'],
                                            self.root.state.get('height', 8),
                                            self.env.Bits)

                # Policy cuando mejora
                if self.max_reward_ever > self.best_reward_logged:
                    self.best_reward_logged = self.max_reward_ever
                    policy = self.get_policy()
                    self.logger.log_policy_text(it + 1, policy)

        self.logger.close()
        return self._get_final_stats(time.perf_counter() - t_start)

    def _select_and_expand(self, C):
        """
        SELECTION mejorada: si enable_inverse_transport está activo,
        desciende por media de recompensas en lugar de UCT.
        """
        node = self.root
        depth = 0

        # Descender
        while node.is_fully_expanded and node.children:
            depth += 1

            if self.enable_inverse_transport:
                # Transporte inverso: descender por mejores rewards
                node = node.best_child_for_selection(method='mean')
            else:
                # UCT estándar
                node = node.best_child_for_expansion(C)

            if node is None:
                break

        # Expandir
        if node and node.untried_actions:
            child, _, _ = node.expand(self.env)
            depth += 1
            return child, depth

        return None, depth

    def _simulate(self, node):
        """SIMULATION: lanza rollouts en GPU."""
        rewards = self.cuda_env.rollout_from_state(
            node.state,
            n_rollouts=self.n_rollouts,
            rollout_depth=self.env.CC,
        )

        done_mask = self.cuda_env.done[:self.n_rollouts]

        if not done_mask.any():
            return -10.0

        done_rewards = rewards[done_mask]
        return float(done_rewards.mean().item())

    def _compute_intrinsic_bonus(self, node, current_it, total_iters):
        """
        Bonificación auxiliar para favorecer exploración temprana.

        Early stage (0-30%): reward si el nodo tiene muchas opciones aún
        Late stage (70-100%): reward si el nodo tiene alta recompensa

        Beneficio: evita quedarse atrapado en caminos malos.
        """
        progress = current_it / max(total_iters, 1)

        if progress < 0.3:
            # Fase temprana: favorecer opciones disponibles
            options_left = len(node.untried_actions)
            total_options = len(self.env.possible_actions)
            bonus = 0.5 * (options_left / total_options)
        elif progress > 0.7:
            # Fase tardía: favorecer nodos que convergen
            bonus = 0.2 * max(0, node.avg_reward / 10.0)
        else:
            # Fase media: transición
            bonus = 0.1

        return bonus

    def _backpropagate(self, node, reward, intrinsic_bonus=0.0):
        """BACKPROPAGATION: sube actualizando con reward + bonus."""
        while node is not None:
            node.update(reward, intrinsic_bonus)
            node = node.parent

    def _print_summary(self, current, total, t_start, C):
        """Resumen periódico."""
        elapsed = time.perf_counter() - t_start
        n_nodes = self._count_nodes()
        avg_reward = np.mean(self._rewards[-100:])
        avg_depth = np.mean(self._selection_depths[-100:])

        child_visits = [c.visits for c in self.root.children]
        max_v = max(child_visits) if child_visits else 0
        min_v = min(child_visits) if child_visits else 0

        print(f"  [{current:>5}/{total}]  "
              f"t={elapsed:6.1f}s  "
              f"C={C:6.3f}  "
              f"nodes={n_nodes:>5}  "
              f"depth={avg_depth:5.1f}  "
              f"r_avg={avg_reward:>7.2f}  "
              f"r_max={self.max_reward_ever:>7.2f}  "
              f"v:[{min_v:>4},{max_v:>4}]")

    def _get_final_stats(self, total_time):
        """Estadísticas finales."""
        return {
            'iteraciones': self.iteration + 1,
            'nodos': self._count_nodes(),
            'profundidad': self._max_depth(),
            'mejor_reward': round(self.max_reward_ever, 4),
            'reward_promedio': round(np.mean(self._rewards), 4),
            'tiempo_total': round(total_time, 2),
        }

    def get_policy(self):
        """Extrae política del árbol."""
        policy = []
        node = self.root

        while node.children:
            best = max(node.children, key=lambda c: c.visits)
            policy.append({
                'step': len(policy),
                'action_idx': best.action,
                'action_name': self.env.possible_actions[best.action],
                'visits': best.visits,
                'reward_mean': best.avg_reward,
                'quality': best.quality_score,
            })
            node = best

        return policy

    def print_policy(self):
        """Imprime política."""
        policy = self.get_policy()

        print(f"\n{'='*100}")
        print(f"  📋 POLÍTICA EXTRAÍDA (v2)")
        print(f"{'='*100}\n")

        if not policy:
            print("  (Sin política)")
            return

        print(f"  {'Celda':>6}  {'Acción':<32}  {'Visitas':>8}  {'R_medio':>8}  {'Quality':>8}")
        print("  " + "-"*100)

        for p in policy[:min(len(policy), self.env.CC)]:
            print(f"  [{p['step']:>2}]"
                  f"  {p['action_name']:<32}"
                  f"  {p['visits']:>8}"
                  f"  {p['reward_mean']:>8.2f}"
                  f"  {p['quality']:>8.2f}")

        print("=" * 100)

    def print_convergence_phases(self):
        """Análisis de convergencia por fases."""
        print(f"\n{'='*80}")
        print(f"  📊 CONVERGENCIA POR FASES")
        print(f"{'='*80}\n")

        if len(self._rewards) < 3:
            return

        rewards = np.array(self._rewards)
        n = len(rewards)
        early = rewards[:n//3]
        mid = rewards[n//3:2*n//3]
        late = rewards[2*n//3:]

        for label, phase in [("Temprana", early), ("Media", mid), ("Tardía", late)]:
            print(f"  {label:>10}:  "
                  f"r_mean={phase.mean():>8.2f}  "
                  f"r_std={phase.std():>7.2f}  "
                  f"r_max={phase.max():>8.2f}  "
                  f"iters={len(phase):>5}")

        improvement = rewards[-100:].mean() - rewards[:100].mean()
        print(f"\n  Mejora (últimas 100 vs primeras 100): {improvement:>8.2f}")
        print("=" * 80)

    # =========================================================================
    # Utilidades
    # =========================================================================

    def _count_nodes(self, node=None):
        if node is None:
            node = self.root
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _max_depth(self, node=None):
        if node is None:
            node = self.root
        if not node.children:
            return 1
        return 1 + max(self._max_depth(c) for c in node.children)


# =============================================================================
# MAIN - Comparación
# =============================================================================

def main():
    """
    Ejemplo: ejecuta MCTS v2 y muestra análisis.
    """
    BITS = 2
    HEIGHT = 2
    ITERATIONS = 20000
    N_ROLLOUTS = 512

    print("\n" + "="*90)
    print("  MCTS + UCT v2: Mejoras Progresivas")
    print("="*90)

    # Crear entornos
    env = BinaryMathEnv(Bits=BITS, Proof=(2**BITS)**2, height=HEIGHT)
    env.reset()

    cuda_env = BinaryMathEnvCUDA(
        Bits=BITS, height=HEIGHT,
        n_envs=N_ROLLOUTS,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Ejecutar con mejoras
    mcts = MCTSWithUCTV2(
        env=env,
        cuda_env=cuda_env,
        n_rollouts=N_ROLLOUTS,
        C_init=1.41,
        enable_inverse_transport=True,
        enable_dynamic_C=True,
        enable_intrinsic_reward=True,
        verbose=True,
    )

    stats = mcts.search(iterations=ITERATIONS, log_every=100)

    # Resultados
    print("\n" + "="*90)
    print("  📈 ESTADÍSTICAS")
    print("="*90 + "\n")
    for k, v in stats.items():
        print(f"  {k:<35} {v}")

    mcts.print_policy()
    mcts.print_convergence_phases()

    print("\n" + "="*90)
    print("  ✓ Búsqueda completada")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
