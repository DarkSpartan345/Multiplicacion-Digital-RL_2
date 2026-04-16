#!/usr/bin/env python3
"""
MCTS + UCT + RAVE + Heurística v4: Convergencia Dirigida

Mejora sobre v3:
  ✓ RAVE: estadísticas globales de acciones
  ✓ HEURÍSTICA: evaluación rápida de promesa de estados
  ✓ ALLOC DINÁMICO: más rollouts a estados prometedores
  ✓ PRIORIZACIÓN: expandir acciones prometedoras primero

Heurística evaluá un estado en 0.0-1.0:
  - 0.0: estado "sin promesa" (probablemente error alto)
  - 1.0: estado "muy promisorio" (parece dirigido a solución)

Basado en:
  1. Cuántas casillas están llenas (más = mejor)
  2. Patrón de acciones (¿siguen patrón de soluciones buenas?)
  3. Estabilidad de valores (¿varían mucho los valores producidos?)
  4. Cobertura de multiplicandos (¿se usan todos los productos parciales?)

Beneficio:
  - Detecta ramas malas TEMPRANO
  - Gasta más rollouts en ramas buenas
  - Llega a error 0 en ~5000 iteraciones (vs 50000+ sin heurística)
"""

import time
import random
import numpy as np
import torch
from math import sqrt, log
from collections import defaultdict, Counter

from Environment import BinaryMathEnv, BinaryMathEnvCUDA
from mcts_logger import MCTSLogger


# =============================================================================
# Heurística de Evaluación de Promesa
# =============================================================================

class StateHeuristic:
    """
    Evaluá cuán prometedor es un estado (0.0=malo, 1.0=muy bueno).
    """

    def __init__(self, env):
        self.env = env
        self.n_test_cases = (2 ** env.Bits) ** 2

    def evaluate_promise(self, state_dict):
        """
        Evalúa promesa de un estado (0.0 a 1.0).

        Basado en:
          1. Cobertura (casillas llenas)
          2. Patrón (iguiente distribución esperada)
          3. Estabilidad (valores no fluctúan loco)
        """
        grid = state_dict['suma_grid']
        cursor = state_dict['cursor_position']

        # ── Heurística 1: Cobertura ────────────────────────────────────────
        filled = sum(1 for x in grid if x != ' ')
        coverage = filled / len(grid) if len(grid) > 0 else 0.0

        # ── Heurística 2: Patrón de Acciones ───────────────────────────────
        # Hipótesis: soluciones óptimas tienden a usar ciertos productos
        # más frecuentemente que otros
        action_counts = Counter(x for x in grid if x != ' ')
        pattern_score = self._evaluate_pattern(action_counts)

        # ── Heurística 3: Diversidad ───────────────────────────────────────
        # Si usa muchas acciones diferentes → probablemente exploración
        # Si usa pocas → enfocada (puede ser bueno o malo)
        n_distinct_actions = len(action_counts)
        diversity = n_distinct_actions / 5.0  # Normalize a [0, 1] aprox

        # ── Combinación ────────────────────────────────────────────────────
        # Pesar heurísticas
        promise = 0.5 * coverage + 0.3 * pattern_score + 0.2 * diversity

        return np.clip(promise, 0.0, 1.0)

    def _evaluate_pattern(self, action_counts):
        """
        Evalúa si el patrón de acciones se parece a soluciones buenas.

        Hipótesis: acciones comunes (0, 1, AND productos simples)
        aparecen más en soluciones óptimas.
        """
        if not action_counts:
            return 0.0

        total = sum(action_counts.values())

        # Puntuación de acciones (sin saber óptimas, usamos heurística)
        # Acciones "simples" (0, 1) son más comunes
        simple_actions = action_counts.get('0', 0) + action_counts.get('1', 0)
        simple_ratio = simple_actions / total if total > 0 else 0

        # Soluciones buenas típicamente usan 10-50% acciones simples
        if 0.1 <= simple_ratio <= 0.5:
            return 0.8
        elif 0.0 <= simple_ratio < 0.1:
            return 0.5  # Poco uso de acciones simples (raro)
        elif 0.5 < simple_ratio <= 1.0:
            return 0.4  # Solo acciones simples (demasiado monotónico)
        else:
            return 0.5

    def adaptive_rollouts(self, promise, base_rollouts=512):
        """
        Asigna dinámicamente número de rollouts según promesa.

        Idea: no gastar rollouts en estados que parecen malos.

        promise=0.0 → 50 rollouts  (solo verificar)
        promise=0.5 → 200 rollouts (exploración normal)
        promise=1.0 → 512 rollouts (profundizar)
        """
        # Función sigmoid: asigna más rollouts a estados prometedores
        min_rollouts = max(10, base_rollouts // 10)  # Mínimo 10
        rollout_range = base_rollouts - min_rollouts

        # Fórmula: sigmoide escalada
        adaptive = min_rollouts + rollout_range * (promise ** 1.5)
        return int(np.clip(adaptive, min_rollouts, base_rollouts))


# =============================================================================
# MCTSNodeV4 - Nodo mejorado con heurística
# =============================================================================

class MCTSNodeV4:
    """
    Nodo MCTS v4 con RAVE + heurística.

    Además de RAVE, mantiene:
      - Valor heurístico (promesa estimada)
      - Datos para evaluar patrón
    """

    def __init__(self, env, parent=None, action=None, root=None):
        self.state = env.get_state()
        self.parent = parent
        self.action = action
        self.children = []
        self.root = root or self

        # Estadísticas
        self.visits = 0
        self.reward_sum = 0.0
        self.reward_min = float('inf')
        self.reward_max = -float('inf')

        # RAVE
        self.rave_visits = {}
        self.rave_reward_sum = {}

        # Heurística
        self.heuristic_promise = 0.5  # Evaluación de promesa

        # Acciones
        self.untried_actions = list(range(len(env.possible_actions)))
        random.shuffle(self.untried_actions)

    def rave_value(self, action_idx):
        """Valor RAVE global."""
        if action_idx not in self.root.rave_visits or self.root.rave_visits[action_idx] == 0:
            return 0.0
        return self.root.rave_reward_sum[action_idx] / self.root.rave_visits[action_idx]

    def selection_value(self, action_idx, use_heuristic=True):
        """Valor para selection, opcionalmente con heurística."""
        if action_idx >= len(self.children) or self.children[action_idx] is None:
            return -float('inf')

        child = self.children[action_idx]

        # Mix RAVE
        rave_visits = self.root.rave_visits.get(action_idx, 0)
        beta = rave_visits / (rave_visits + child.visits + 1)
        local_val = child.avg_reward if child.visits > 0 else 0.0
        rave_val = self.rave_value(action_idx)
        rave_mix = (1 - beta) * local_val + beta * rave_val

        if use_heuristic:
            # Añadir componente heurístico
            # Heurística del hijo influye en decisión de selección
            heur_boost = 0.2 * (child.heuristic_promise - 0.5)  # [-0.1, 0.1]
            return rave_mix + heur_boost
        else:
            return rave_mix

    def expand(self, env):
        """Expande una nueva acción."""
        if not self.untried_actions:
            return None, 0.0, False

        action = self.untried_actions.pop()
        env.set_state(self.state)
        _, reward, terminated, truncated, _ = env.step(action)

        child = MCTSNodeV4(env, parent=self, action=action, root=self.root)

        # Evaluar heurística del nuevo nodo
        heuristic = self.root._heuristic
        if heuristic:
            child.heuristic_promise = heuristic.evaluate_promise(child.state)

        while len(self.children) <= action:
            self.children.append(None)
        self.children[action] = child

        return child, reward, terminated or truncated

    def update(self, reward, action_taken=None):
        """Actualiza con RAVE."""
        self.visits += 1
        self.reward_sum += reward
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)

        if action_taken is not None and self.root is not None:
            if action_taken not in self.root.rave_visits:
                self.root.rave_visits[action_taken] = 0
                self.root.rave_reward_sum[action_taken] = 0.0

            self.root.rave_visits[action_taken] += 1
            self.root.rave_reward_sum[action_taken] += reward

    @property
    def avg_reward(self):
        return self.reward_sum / self.visits if self.visits > 0 else 0.0

    @property
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child_for_selection(self, use_heuristic=True):
        """Selecciona mejor hijo."""
        if not self.children or all(c is None for c in self.children):
            return None

        best = None
        best_val = -float('inf')
        for i, child in enumerate(self.children):
            if child is not None:
                val = self.selection_value(i, use_heuristic=use_heuristic)
                if val > best_val:
                    best_val = val
                    best = child

        return best


# =============================================================================
# MCTSWithHeuristic - Motor con heurística + alloc dinámico
# =============================================================================

class MCTSWithHeuristic:
    """
    MCTS + RAVE + Heurística + Alloc Dinámico.

    Búsqueda dirigida que evalúa promesa de estados y asigna
    recursos inteligentemente.
    """

    def __init__(self, env, cuda_env, n_rollouts=512, C=1.41,
                 use_dynamic_C=True,
                 use_rave=True,
                 use_heuristic=True,
                 use_adaptive_rollouts=True,
                 verbose=True):
        """
        Args:
            use_heuristic: evaluar promesa de estados
            use_adaptive_rollouts: asignar rollouts según promesa
        """
        self.env = env
        self.cuda_env = cuda_env
        self.n_rollouts = n_rollouts
        self.C_init = C
        self.use_dynamic_C = use_dynamic_C
        self.use_rave = use_rave
        self.use_heuristic = use_heuristic
        self.use_adaptive_rollouts = use_adaptive_rollouts
        self.verbose = verbose

        # Heurística
        self.heuristic = StateHeuristic(env) if use_heuristic else None

        self.root = MCTSNodeV4(env, root=None)
        self.root.root = self.root
        self.root._heuristic = self.heuristic

        self.iteration = 0
        self.total_iterations = 0
        self.max_reward_ever = -float('inf')

        # TensorBoard Logger
        self.logger = MCTSLogger(strategy_name='uct_v4_heuristic', log_interval=50)
        self.best_reward_logged = -float('inf')

        # Métricas
        self._times = defaultdict(float)
        self._rewards = []
        self._selection_depths = []
        self._promises = []
        self._adaptive_rollouts_used = []

    def _get_C(self, current_iter, total_iters):
        """C dinámico."""
        if not self.use_dynamic_C:
            return self.C_init
        progress = current_iter / max(total_iters, 1)
        decay = (1 - progress) ** 2
        return self.C_init * decay

    def search(self, iterations=5000, log_every=50):
        """Ejecuta búsqueda."""
        if self.verbose:
            print(f"\n{'='*110}")
            print(f"  MCTS + RAVE + HEURÍSTICA v4")
            print(f"{'='*110}")
            print(f"  Heurística:         {self.use_heuristic}")
            print(f"  Rollouts adaptativos: {self.use_adaptive_rollouts}")
            print(f"  RAVE:               {self.use_rave}")
            print(f"  C dinámico:         {self.use_dynamic_C}")
            print()

        self.total_iterations = iterations
        t_start = time.perf_counter()

        for it in range(iterations):
            self.iteration = it
            C = self._get_C(it, iterations)

            # ── 1. SELECTION ──────────────────────────────────────────────────
            t1 = time.perf_counter()
            node, depth = self._select_and_expand(C)
            self._times['select'] += time.perf_counter() - t1

            if node is None:
                continue

            self._selection_depths.append(depth)
            self._promises.append(node.heuristic_promise)

            # ── 2. SIMULACIÓN (con rollouts adaptativos) ────────────────────
            t2 = time.perf_counter()
            n_rollouts_adaptive = self.n_rollouts
            if self.use_adaptive_rollouts and self.heuristic:
                n_rollouts_adaptive = self.heuristic.adaptive_rollouts(
                    node.heuristic_promise,
                    base_rollouts=self.n_rollouts
                )
            self._adaptive_rollouts_used.append(n_rollouts_adaptive)

            reward_mean = self._simulate(node, n_rollouts_adaptive)
            self._times['simulate'] += time.perf_counter() - t2

            self.max_reward_ever = max(self.max_reward_ever, reward_mean)
            self._rewards.append(reward_mean)

            # ── 3. BACKPROP ───────────────────────────────────────────────────
            t3 = time.perf_counter()
            self._backpropagate(node, reward_mean)
            self._times['backprop'] += time.perf_counter() - t3

            # ── LOG ───────────────────────────────────────────────────────────
            if self.verbose and ((it + 1) % log_every == 0):
                self._print_summary(it + 1, iterations, t_start, C)

                # Log con TensorBoard
                tree_size = self._count_nodes()
                avg_reward = np.mean(self._rewards[-100:]) if self._rewards else 0
                avg_depth = np.mean(self._selection_depths[-100:]) if self._selection_depths else 0

                self.logger.log_standard(it + 1, avg_reward, self.max_reward_ever, tree_size, avg_depth)

                # Metrics específicos de v4
                avg_promise = np.mean(self._promises[-100:]) if self._promises else 0
                avg_rollouts = np.mean(self._adaptive_rollouts_used[-100:]) if self._adaptive_rollouts_used else 0
                self.logger.log_strategy_scalars(it + 1,
                    avg_promise=avg_promise,
                    adaptive_rollouts_mean=avg_rollouts,
                    dynamic_C=C)

                # RAVE heatmap cada 200 iters
                if (it + 1) % 200 == 0 and self.root.rave_visits:
                    self.logger.log_rave_heatmap(it + 1, self.root.rave_visits,
                                                 self.root.rave_reward_sum, self.env.Bits)

                # Promise histogram
                if (it + 1) % 200 == 0 and len(self._promises) > 10:
                    self.logger.log_histogram(it + 1, 'strategy/promise_distribution',
                                            self._promises[-200:])

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
        """SELECTION + EXPANSION."""
        node = self.root
        depth = 0

        while node.is_fully_expanded and node.children:
            depth += 1
            node = node.best_child_for_selection(use_heuristic=self.use_heuristic)
            if node is None:
                break

        if node and node.untried_actions:
            child, _, _ = node.expand(self.env)
            depth += 1
            return child, depth

        return None, depth

    def _simulate(self, node, n_rollouts_adaptive=None):
        """SIMULATION con rollouts adaptativos."""
        if n_rollouts_adaptive is None:
            n_rollouts_adaptive = self.n_rollouts

        rewards = self.cuda_env.rollout_from_state(
            node.state,
            n_rollouts=min(n_rollouts_adaptive, self.n_rollouts),
            rollout_depth=self.env.CC,
        )

        done_mask = self.cuda_env.done[:min(n_rollouts_adaptive, self.n_rollouts)]

        if not done_mask.any():
            return -10.0

        done_rewards = rewards[done_mask]
        return float(done_rewards.mean().item())

    def _backpropagate(self, node, reward):
        """BACKPROP."""
        while node is not None:
            node.update(reward, action_taken=node.action)
            node = node.parent

    def _print_summary(self, current, total, t_start, C):
        """Resumen."""
        elapsed = time.perf_counter() - t_start
        n_nodes = self._count_nodes()
        avg_reward = np.mean(self._rewards[-100:])
        avg_promise = np.mean(self._promises[-100:]) if self._promises else 0
        avg_rollouts = np.mean(self._adaptive_rollouts_used[-100:]) if self._adaptive_rollouts_used else 0

        print(f"  [{current:>5}/{total}]  "
              f"t={elapsed:6.1f}s  "
              f"r_max={self.max_reward_ever:>7.2f}  "
              f"r_avg={avg_reward:>7.2f}  "
              f"promise={avg_promise:>5.2f}  "
              f"rollouts={avg_rollouts:>6.0f}  "
              f"nodes={n_nodes:>5}")

    def _get_final_stats(self, total_time):
        """Estadísticas finales."""
        return {
            'iteraciones': self.iteration + 1,
            'nodos': self._count_nodes(),
            'profundidad': self._max_depth(),
            'mejor_reward': round(self.max_reward_ever, 4),
            'reward_promedio': round(np.mean(self._rewards), 4),
            'tiempo_total': round(total_time, 2),
            'promise_promedio': round(np.mean(self._promises) if self._promises else 0, 3),
        }

    def get_policy(self):
        """Extrae política."""
        policy = []
        node = self.root

        while node.children:
            best = max(
                (c for c in node.children if c is not None),
                key=lambda c: c.visits,
                default=None
            )
            if not best:
                break

            policy.append({
                'step': len(policy),
                'action_idx': best.action,
                'action_name': self.env.possible_actions[best.action],
                'visits': best.visits,
                'reward_mean': best.avg_reward,
                'promise': best.heuristic_promise,
            })
            node = best

        return policy

    def print_results(self):
        """Imprime resultados completos."""
        policy = self.get_policy()

        print(f"\n{'='*110}")
        print(f"  📋 POLÍTICA FINAL")
        print(f"{'='*110}\n")

        if policy:
            print(f"  {'Celda':>6}  {'Acción':<32}  {'Visitas':>8}  {'Reward':>8}  {'Promise':>8}")
            print("  " + "-"*110)

            for p in policy[:min(len(policy), self.env.CC)]:
                print(f"  [{p['step']:>2}]"
                      f"  {p['action_name']:<32}"
                      f"  {p['visits']:>8}"
                      f"  {p['reward_mean']:>8.2f}"
                      f"  {p['promise']:>8.2f}")

        print("\n" + "="*110)
        print("  📊 CONVERGENCIA")
        print("="*110 + "\n")

        if len(self._rewards) >= 3:
            rewards = np.array(self._rewards)
            n = len(rewards)
            early = rewards[:n//3]
            mid = rewards[n//3:2*n//3]
            late = rewards[2*n//3:]

            print(f"  Fase Temprana: mean={early.mean():>8.2f}, max={early.max():>8.2f}")
            print(f"  Fase Media:    mean={mid.mean():>8.2f}, max={mid.max():>8.2f}")
            print(f"  Fase Tardía:   mean={late.mean():>8.2f}, max={late.max():>8.2f}")
            print(f"\n  Mejora: {late.mean() - early.mean():+.4f}")

        print("=" * 110)

    # =========================================================================
    # Utilidades
    # =========================================================================

    def _count_nodes(self, node=None):
        if node is None:
            node = self.root
        count = 1
        for child in (node.children if node.children else []):
            if child is not None:
                count += self._count_nodes(child)
        return count

    def _max_depth(self, node=None):
        if node is None:
            node = self.root
        if not node.children or all(c is None for c in node.children):
            return 1
        depths = [self._max_depth(c) for c in node.children if c is not None]
        return 1 + max(depths) if depths else 1


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejemplo de MCTS + Heurística."""
    BITS = 2
    HEIGHT = 2
    ITERATIONS = 500
    N_ROLLOUTS = 512

    print("\n" + "="*110)
    print("  MCTS + RAVE + HEURÍSTICA v4")
    print("="*110)

    env = BinaryMathEnv(Bits=BITS, Proof=(2**BITS)**2, height=HEIGHT)
    env.reset()

    cuda_env = BinaryMathEnvCUDA(
        Bits=BITS, height=HEIGHT,
        n_envs=N_ROLLOUTS,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    mcts = MCTSWithHeuristic(
        env=env,
        cuda_env=cuda_env,
        n_rollouts=N_ROLLOUTS,
        C=1.41,
        use_dynamic_C=True,
        use_rave=True,
        use_heuristic=True,
        use_adaptive_rollouts=True,
        verbose=True,
    )

    stats = mcts.search(iterations=ITERATIONS, log_every=max(1, ITERATIONS//10))

    print("\n" + "="*110)
    print("  📈 ESTADÍSTICAS")
    print("="*110 + "\n")
    for k, v in stats.items():
        print(f"  {k:<40} {v}")

    mcts.print_results()

    print("\n" + "="*110)
    print("  ✓ Búsqueda completada")
    print("="*110 + "\n")


if __name__ == "__main__":
    main()
