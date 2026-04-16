"""
Entorno CUDA/batch para MCTS paralelo.

Reemplaza la compilación Verilog (iverilog + vvp) con operaciones PyTorch
vectorizadas, permitiendo evaluar miles de entornos simultáneamente en GPU.

Diferencias clave vs. BinaryMathEnv:
- Sin dependencia de iverilog/vvp
- N entornos en paralelo como tensores
- Evaluación de la tabla en microsegundos (vs. ~100ms por compilación Verilog)
- Compatible con el mismo espacio de acciones que BinaryMathEnv

Uso típico en MCTS:
    cuda_env = BinaryMathEnvCUDA(Bits=2, height=2, n_envs=2048, device='cuda')
    rewards = cuda_env.rollout_from_state(node.state, n_rollouts=2048)

Modo incremental (incremental=True):
    Llenado columna a columna (LSB → MSB). Cada columna completada emite reward
    inmediato basado en XOR ponderado entre bits calculados y esperados.
    Permite a MCTS detectar mejoras parciales sin esperar al final del episodio.
"""

import torch
import numpy as np


class BinaryMathEnvCUDA:
    """
    Versión CUDA/batched de BinaryMathEnv.

    Corre N entornos en paralelo. El espacio de acciones es idéntico al
    de BinaryMathEnv para que ambos sean intercambiables en MCTS.

    La evaluación del circuito replica la lógica Verilog:
      1. Para cada columna de la tabla, recoge los productos únicos presentes.
      2. Suma sus valores bit a bit para A, B dados.
      3. P = suma_col(col_sum << shift_col), igual que el assign en Verilog.

    Con incremental=True:
      - El cursor llena columnas en orden LSB → MSB (column-major).
      - Al completar cada columna se computa sum_bit + carry_out y se emite
        reward ponderado por posición de bit.
      - Rango de reward incremental: [-10, 0] (circuito perfecto → 0).
    """

    def __init__(self, Bits=8, Proof=4, height=8, n_envs=1024, device='cuda',
                 incremental=False):
        """
        Args:
            Bits:        Número de bits del multiplicador.
            Proof:       Ignorado (se usan todos los pares exhaustivos).
            height:      Filas de la tabla de productos parciales.
            n_envs:      Número de entornos en paralelo.
            device:      'cuda' o 'cpu'.
            incremental: Si True, usa llenado column-major LSB-first con reward
                         incremental por columna. Si False, comportamiento original.
        """
        self.Bits    = Bits
        self.Proof   = Proof
        self.height  = height
        self.n_envs  = n_envs
        self.device  = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.incremental = incremental

        self.CC        = height * 2 * Bits   # total casillas por entorno
        self.grid_size = 2 * Bits            # columnas de la tabla

        # Construir espacio de acciones (mismo orden que BinaryMathEnv)
        self._build_action_space()
        # Pre-calcular tablas de decodificación en GPU
        self._build_decode_tables()

        # ── Tensores de estado ────────────────────────────────────────────────
        # suma_grid:   (n_envs, CC)   — índice de acción por celda, -1 = vacío
        # cursor_pos:  (n_envs,)      — siguiente celda a rellenar
        # done:        (n_envs,)      — True si el episodio terminó
        # rewards:     (n_envs,)      — recompensa acumulada del episodio
        self.suma_grid  = torch.full((n_envs, self.CC), -1,
                                     dtype=torch.int16, device=self.device)
        self.cursor_pos = torch.zeros(n_envs, dtype=torch.int32, device=self.device)
        self.done       = torch.zeros(n_envs, dtype=torch.bool,  device=self.device)
        self.rewards    = torch.zeros(n_envs, dtype=torch.float32, device=self.device)

        # Casos de prueba exhaustivos: TODOS los pares (A, B) posibles
        self._build_exhaustive_test_cases()

        # ── Tensores para modo incremental ───────────────────────────────────
        if self.incremental:
            # carry_in: carry acumulado entre evaluaciones de columna
            # (n_envs, n_test_cases) int32 — bounded por height (~8), cabe en int32
            self.carry_in = torch.zeros(n_envs, self.n_test_cases,
                                        dtype=torch.int32, device=self.device)
            # Logging opcional (activar con enable_column_logging())
            self._logging_enabled = False
            self._col_error_log   = None   # (n_envs, grid_size) float32
            self._carry_log       = None   # (n_envs, grid_size+1) int32

    # =========================================================================
    # Construcción interna
    # =========================================================================

    def _build_action_space(self):
        """Mismo orden que BinaryMathEnv._generate_possible_actions()."""
        self.possible_actions = ['0', '1']
        for i in range(self.Bits):
            for j in range(self.Bits):
                self.possible_actions.append(f'(A[{i}]&B[{j}])')
                self.possible_actions.append(f'(~A[{i}]&B[{j}])')
                self.possible_actions.append(f'(A[{i}]&~B[{j}])')
                self.possible_actions.append(f'(~A[{i}]&~B[{j}])')
        self.n_actions = len(self.possible_actions)
        # Mapa inverso: string → índice (para convertir estados de BinaryMathEnv)
        self._action_to_idx = {a: i for i, a in enumerate(self.possible_actions)}

    def _build_decode_tables(self):
        """
        Pre-calcula en GPU la decodificación de cada acción:
          action_i_bits[k]  — bit de A usado por el k-ésimo producto parcial
          action_j_bits[k]  — bit de B
          action_neg_a[k]   — True si A está negado
          action_neg_b[k]   — True si B está negado
        Sólo para acciones >= 2 (las parciales). Las acciones 0 y 1 se manejan
        directamente en _evaluate_all_actions.
        """
        n_pp = self.n_actions - 2
        i_bits = torch.empty(n_pp, dtype=torch.int64)
        j_bits = torch.empty(n_pp, dtype=torch.int64)
        neg_a  = torch.empty(n_pp, dtype=torch.bool)
        neg_b  = torch.empty(n_pp, dtype=torch.bool)

        k = 0
        for i in range(self.Bits):
            for j in range(self.Bits):
                # orden: A&B, ~A&B, A&~B, ~A&~B
                i_bits[k]=i; j_bits[k]=j; neg_a[k]=False; neg_b[k]=False; k+=1
                i_bits[k]=i; j_bits[k]=j; neg_a[k]=True;  neg_b[k]=False; k+=1
                i_bits[k]=i; j_bits[k]=j; neg_a[k]=False; neg_b[k]=True;  k+=1
                i_bits[k]=i; j_bits[k]=j; neg_a[k]=True;  neg_b[k]=True;  k+=1

        self.action_i_bits = i_bits.to(self.device)
        self.action_j_bits = j_bits.to(self.device)
        self.action_neg_a  = neg_a.to(self.device)
        self.action_neg_b  = neg_b.to(self.device)

        # Factor de escala por columna (replica el << del Verilog)
        # col=0 → shift = grid_size-1, col=grid_size-1 → shift=0
        shifts = torch.tensor(
            [self.grid_size - col - 1 for col in range(self.grid_size)],
            dtype=torch.float64
        )
        self.shift_factors = (2.0 ** shifts).to(self.device)   # (grid_size,)

    def _build_exhaustive_test_cases(self):
        """
        Genera TODOS los casos posibles de multiplicacion (A, B) para Bits dado.
        Para Bits=2: A in {0,1,2,3}, B in {0,1,2,3} → 16 casos.
        Para Bits=4: 256 casos. Para Bits=8: 65536 casos.

        Los test cases son identicos para todos los entornos (determinista).
        Se almacenan como tensores (n_envs, n_cases) donde cada entorno tiene
        exactamente los mismos valores.
        """
        max_val = 2 ** self.Bits  # 0..max_val-1
        self.n_test_cases = max_val * max_val

        # Crear todos los pares (a, b) con meshgrid
        a_range = torch.arange(max_val, dtype=torch.int64, device=self.device)
        b_range = torch.arange(max_val, dtype=torch.int64, device=self.device)
        grid_a, grid_b = torch.meshgrid(a_range, b_range, indexing='ij')

        # Aplanar a vectores (n_cases,)
        self._all_A = grid_a.flatten()   # (n_cases,)
        self._all_B = grid_b.flatten()   # (n_cases,)
        self._all_true_P = self._all_A * self._all_B  # (n_cases,)

        # Maximo producto posible (para normalizar error)
        self.max_product = (max_val - 1) ** 2  # (2^Bits - 1)^2

        # Expandir a (n_envs, n_cases) — todos los entornos usan los mismos casos
        self.A_vals = self._all_A.unsqueeze(0).expand(self.n_envs, -1)
        self.B_vals = self._all_B.unsqueeze(0).expand(self.n_envs, -1)
        self.true_products = self._all_true_P.unsqueeze(0).expand(self.n_envs, -1)

        # Actualizar Proof para reflejar el numero real de test cases
        self.Proof = self.n_test_cases

    # =========================================================================
    # API principal
    # =========================================================================

    def reset(self, env_indices=None):
        """
        Reinicia entornos. Si env_indices=None reinicia todos.
        Los test cases NO se regeneran (son exhaustivos y fijos).

        Args:
            env_indices: lista/tensor de índices, o None para todos.
        """
        if env_indices is None:
            self.suma_grid.fill_(-1)
            self.cursor_pos.fill_(0)
            self.done.fill_(False)
            self.rewards.fill_(0.0)
            if self.incremental:
                self.carry_in.fill_(0)
        else:
            idx = torch.as_tensor(env_indices, device=self.device)
            self.suma_grid[idx]  = -1
            self.cursor_pos[idx] = 0
            self.done[idx]       = False
            self.rewards[idx]    = 0.0
            if self.incremental:
                self.carry_in[idx] = 0

    def step(self, actions):
        """
        Ejecuta un paso para todos los entornos activos en paralelo.

        En modo incremental: el cursor sigue orden column-major LSB-first.
        Cada vez que se completa una columna se evalúa y se emite reward.

        Args:
            actions: (n_envs,) tensor de int con el índice de acción.

        Returns:
            step_rewards: (n_envs,) recompensa del paso actual.
            done:         (n_envs,) bool, True si el entorno completó la tabla.
        """
        active   = ~self.done
        writable = active & (self.cursor_pos < self.CC)

        step_rewards = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)

        if not writable.any():
            return step_rewards, self.done.clone()

        idx = writable.nonzero(as_tuple=True)[0]

        if self.incremental:
            # Column-major LSB-first: mapear cursor → celda física
            flat_cells = self._cursor_to_flat(self.cursor_pos[idx])
            self.suma_grid[idx, flat_cells] = actions[idx].to(torch.int16)
        else:
            cols = self.cursor_pos[idx]
            self.suma_grid[idx, cols] = actions[idx].to(torch.int16)

        self.cursor_pos[idx] += 1

        if self.incremental:
            # Detectar columnas recién completadas: cursor_pos % height == 0
            col_complete = writable & (self.cursor_pos % self.height == 0)
            if col_complete.any():
                prev_rewards = self.rewards.clone()
                comp_idx     = col_complete.nonzero(as_tuple=True)[0]
                bp_vals      = (self.cursor_pos[comp_idx] // self.height - 1).long()

                for bp in bp_vals.unique():
                    col_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                    col_mask[comp_idx[bp_vals == bp]] = True
                    self._evaluate_column_batch(int(bp.item()), col_mask)

                # step_rewards = delta de reward en este paso
                step_rewards = self.rewards - prev_rewards

            # Marcar como terminados
            just_done = writable & (self.cursor_pos >= self.CC)
            if just_done.any():
                self._penalize_overflow(just_done)
                self.done[just_done] = True

        else:
            # Modo original: evaluar solo al terminar
            just_done = writable & (self.cursor_pos >= self.CC)
            if just_done.any():
                r = self._evaluate_batch(just_done)
                self.rewards[just_done] = r
                step_rewards[just_done] = r
                self.done[just_done]    = True

        return step_rewards, self.done.clone()

    # =========================================================================
    # Modo incremental: mapeo de cursor y evaluación por columna
    # =========================================================================

    def _cursor_to_flat(self, cursor_pos):
        """
        Column-major LSB-first: cursor_pos → índice plano en suma_grid.

        Orden de llenado:
          cursor 0..height-1       → columna física grid_size-1 (LSB, bit 0)
          cursor height..2*height-1 → columna física grid_size-2 (bit 1)
          cursor k*height..(k+1)*height-1 → columna física grid_size-1-k (bit k)

        Args:
            cursor_pos: LongTensor (n,) — posiciones de cursor (antes del incremento).

        Returns:
            LongTensor (n,) — índices planos en suma_grid.
        """
        bit_pos = cursor_pos // self.height          # columna lógica (0=LSB)
        row     = cursor_pos % self.height
        col     = self.grid_size - 1 - bit_pos       # columna física
        return (row * self.grid_size + col).long()

    def _evaluate_column_batch(self, bit_pos: int, env_mask: torch.Tensor):
        """
        Evalúa una columna recién completada para los entornos indicados.

        Flujo:
          1. Lee las acciones de la columna (height celdas por entorno).
          2. Deduplica: suma solo la primera aparición de cada acción única.
          3. Acumula total = col_sum + carry_in.
          4. Calcula sum_bit = total & 1, carry_out = total >> 1.
          5. Actualiza carry_in ← carry_out.
          6. Reward = -10 × mean_error × weight / norm, acumulado en self.rewards.

        Implementación memory-efficient: loop sobre height (≤8 iters),
        tensor (n, Proof) por iteración. Evita materializar (n, Proof, n_actions).

        Args:
            bit_pos:  Posición de bit de la columna evaluada (0=LSB).
            env_mask: (n_envs,) BoolTensor — entornos a procesar.
        """
        idx = env_mask.nonzero(as_tuple=True)[0]
        n   = idx.shape[0]
        if n == 0:
            return

        col_idx  = self.grid_size - 1 - bit_pos          # columna física
        grids    = self.suma_grid[idx].long()             # (n, CC)
        grid_3d  = grids.view(n, self.height, self.grid_size)
        col_act  = grid_3d[:, :, col_idx]                 # (n, height)
        valid    = col_act >= 0                           # (n, height) bool
        col_safe = col_act.clamp(min=0).long()            # (n, height) — -1 → 0

        # Compartir los mismos test cases para todos los envs (optimización)
        # Shape: (1, Proof) → broadcast a (n, Proof) en operaciones
        A = self._all_A.unsqueeze(0)    # (1, Proof) int64
        B = self._all_B.unsqueeze(0)    # (1, Proof) int64

        # ── Suma de acciones únicas en la columna (loop sobre height) ─────────
        # col_sum acumula el valor de cada acción única; seen evita duplicados.
        # Pico de memoria: ~3 × (n × Proof) × 4 bytes por iteración.
        col_sum = torch.zeros(n, self.Proof, dtype=torch.float32, device=self.device)
        seen    = torch.zeros(n, self.n_actions, dtype=torch.bool, device=self.device)
        env_idx_arange = torch.arange(n, device=self.device)

        for h in range(self.height):
            act_h  = col_safe[:, h]                          # (n,) long
            val_h  = valid[:, h]                             # (n,) bool
            seen_h = seen[env_idx_arange, act_h]             # (n,) bool
            is_new = val_h & ~seen_h                         # (n,) bool

            if not is_new.any():
                continue

            # Marcar acción como vista para este entorno
            seen[env_idx_arange, act_h] = seen[env_idx_arange, act_h] | is_new

            # Decodificar acción → índices de bit
            pp_idx = (act_h - 2).clamp(min=0)               # (n,) — safe index
            i_bit  = self.action_i_bits[pp_idx]              # (n,) int64
            j_bit  = self.action_j_bits[pp_idx]              # (n,) int64
            na     = self.action_neg_a[pp_idx]               # (n,) bool
            nb     = self.action_neg_b[pp_idx]               # (n,) bool

            # Valor del producto parcial para todos los test cases
            # A >> i_bit[e]: para env e, extrae bit i_bit[e] de todos los A values
            # i_bit: (n,) → unsqueeze → (n, 1) → broadcast con A (1, Proof) → (n, Proof)
            A_bit = ((A >> i_bit.unsqueeze(1)) & 1).float()  # (n, Proof)
            B_bit = ((B >> j_bit.unsqueeze(1)) & 1).float()  # (n, Proof)
            A_eff = torch.where(na.unsqueeze(1), 1.0 - A_bit, A_bit)
            B_eff = torch.where(nb.unsqueeze(1), 1.0 - B_bit, B_bit)
            cell_val = A_eff * B_eff                          # (n, Proof)

            # Constantes: override con valor fijo
            cell_val[act_h == 0] = 0.0   # acción '0' → constante 0
            cell_val[act_h == 1] = 1.0   # acción '1' → constante 1

            # Solo sumar acciones nuevas (deduplicación)
            cell_val[~is_new] = 0.0

            col_sum += cell_val

        # ── Carry + split sum/carry ───────────────────────────────────────────
        carry_in  = self.carry_in[idx].long()                # (n, Proof)
        total     = col_sum.long() + carry_in                # (n, Proof)
        sum_bit   = total & 1                                # (n, Proof)
        carry_out = total >> 1                               # (n, Proof)

        self.carry_in[idx] = carry_out.to(torch.int32)

        # ── Reward ponderado por posición de bit ─────────────────────────────
        # Fórmula DUAL: Reward POSITIVO si columna correcta, NEGATIVO si hay errores
        # - Columna correcta (error_mean=0): reward = +(weight / norm)  [BONUS]
        # - Columna con errores:             reward = -(weight / norm) * error_mean
        true_P    = self._all_true_P.unsqueeze(0)            # (1, Proof) int64
        expected  = (true_P >> bit_pos) & 1                  # (1, Proof) int64
        bit_error = (sum_bit ^ expected).float()             # (n, Proof)
        error_mean = bit_error.mean(dim=1)                   # (n,)

        weight  = float(1 << bit_pos)                        # 2^bit_pos
        norm    = float((1 << self.grid_size) - 1)           # 2^grid_size - 1

        # Reward dual: bonus si error=0, penalización si error>0
        col_reward = torch.where(
            error_mean == 0,
            torch.full_like(error_mean, weight / norm),      # Bonus: columna correcta
            -(weight / norm) * error_mean                     # Penalización por error
        )

        self.rewards[idx] += col_reward

        # ── Logging opcional ─────────────────────────────────────────────────
        if self._logging_enabled:
            self._log_column_stats(bit_pos, bit_error, carry_out, idx)

    def _penalize_overflow(self, env_mask: torch.Tensor):
        """
        Penaliza carry residual no-cero tras evaluar todas las columnas.
        Un multiplicador correcto no tiene overflow; carry ≠ 0 implica error.

        Args:
            env_mask: (n_envs,) BoolTensor.
        """
        idx      = env_mask.nonzero(as_tuple=True)[0]
        overflow = (self.carry_in[idx] > 0).float()   # (n, Proof)
        penalty  = -10.0 * overflow.mean(dim=1)        # (n,)
        self.rewards[idx] += penalty

    def _recompute_carries_silent(self, env_indices):
        """
        Re-evalúa las columnas ya llenas para reconstruir carry_in desde cero.
        No acumula reward (restaura self.rewards después de cada columna).

        Usado por _load_state cuando el estado tiene cursor > 0.

        Args:
            env_indices: lista o tensor de índices de entorno.
        """
        idx_tensor = torch.as_tensor(env_indices, device=self.device)
        self.carry_in[idx_tensor] = 0

        # Número de columnas completadas = cursor_pos // height
        cursor     = int(self.cursor_pos[idx_tensor[0]].item())
        n_cols     = cursor // self.height

        for bp in range(n_cols):
            mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            mask[idx_tensor] = True
            saved_rewards = self.rewards[idx_tensor].clone()
            self._evaluate_column_batch(bp, mask)
            self.rewards[idx_tensor] = saved_rewards   # sin reward

    # =========================================================================
    # Evaluación del circuito (núcleo CUDA — modo original)
    # =========================================================================

    def _evaluate_all_actions(self, A, B):
        """
        Calcula el valor bit (0 ó 1) de cada acción para cada test case.

        Args:
            A: (n, Proof) — valores del operando A.
            B: (n, Proof) — valores del operando B.

        Returns:
            (n, Proof, n_actions) float32  — valor de cada acción.
        """
        n = A.shape[0]
        vals = torch.zeros(n, self.Proof, self.n_actions,
                           dtype=torch.float32, device=self.device)

        # Acción 0 → constante 0 (ya en 0)
        # Acción 1 → constante 1
        vals[:, :, 1] = 1.0

        # Acciones 2.. → productos parciales
        # Extraer el bit i de A para cada acción: (n, Proof, n_pp)
        A_bits = ((A.unsqueeze(-1) >> self.action_i_bits) & 1).float()
        B_bits = ((B.unsqueeze(-1) >> self.action_j_bits) & 1).float()

        # Aplicar negaciones
        A_eff = torch.where(self.action_neg_a, 1.0 - A_bits, A_bits)
        B_eff = torch.where(self.action_neg_b, 1.0 - B_bits, B_bits)

        # AND = multiplicación de bits
        vals[:, :, 2:] = A_eff * B_eff
        return vals   # (n, Proof, n_actions)

    def _compute_products(self, grids, A, B):
        """
        Replica la lógica del Verilog generado por generate_verilog():
          1. Por columna: toma los productos únicos presentes (deduplicación vía max).
          2. Suma sus valores para A, B dados.
          3. P = Σ_col (col_sum << (grid_size - col - 1))

        Args:
            grids: (n, CC)    — índice de acción por celda.
            A:     (n, Proof) — operandos A.
            B:     (n, Proof) — operandos B.

        Returns:
            (n, Proof) int64  — productos computados P.
        """
        n = grids.shape[0]

        # Valores de todas las acciones para estos test cases
        action_vals = self._evaluate_all_actions(A, B)   # (n, Proof, n_actions)

        # Reshape a (n, height, grid_size)
        grid_3d = grids.view(n, self.height, self.grid_size)

        # Máscara de celdas válidas (no vacías)
        valid = (grid_3d >= 0)                             # (n, height, grid_size)

        # Reemplazar -1 por 0 para no romper scatter (se anulará con la máscara)
        grid_safe = grid_3d.clamp(min=0)

        # One-hot → (n, height, grid_size, n_actions)
        one_hot = torch.zeros(n, self.height, self.grid_size, self.n_actions,
                              dtype=torch.float32, device=self.device)
        one_hot.scatter_(-1, grid_safe.long().unsqueeze(-1), 1.0)
        one_hot *= valid.unsqueeze(-1)   # anular celdas vacías

        # Deduplicación: max sobre height → (n, grid_size, n_actions)
        # max actúa como OR lógico: si la acción aparece ≥1 vez en la columna, vale 1
        presence, _ = one_hot.max(dim=1)

        # Suma ponderada por columna: (n, grid_size, Proof)
        # presence[n, col, act] × action_vals[n, proof, act] → sum sobre act
        col_sums = torch.einsum('eca,epa->ecp', presence, action_vals)

        # Aplicar desplazamientos y sumar columnas → (n, Proof)
        # shift_factors[col] = 2^(grid_size - col - 1)
        products = torch.einsum('ecp,c->ep', col_sums.double(), self.shift_factors)

        return products.long()

    def _evaluate_batch(self, env_mask):
        """
        Evalúa los entornos señalados por env_mask.

        Reward = -10 * E[|error|] / max_product
        donde E[|error|] = mean(|P_circuito - P_esperado|) sobre TODOS los
        pares (A, B) posibles.

        - Reward = 0    → circuito perfecto (sin errores)
        - Reward = -10  → peor caso posible (error maximo en todos los casos)

        Args:
            env_mask: (n_envs,) bool.

        Returns:
            (suma(env_mask),) float32 — recompensas en [-10, 0].
        """
        idx       = env_mask.nonzero(as_tuple=True)[0]
        grids     = self.suma_grid[idx].long()
        A         = self.A_vals[idx]
        B         = self.B_vals[idx]
        true_P    = self.true_products[idx].float()

        computed_P = self._compute_products(grids, A, B).float()

        # Error absoluto normalizado por max_product
        error = computed_P - true_P
        abs_error = error.abs().long()

        max_bits = int(torch.ceil(torch.log2(torch.tensor(self.max_product + 1))))
        bit_positions = torch.arange(max_bits, device=abs_error.device, dtype=torch.long)
        bit_errors = ((abs_error.unsqueeze(-1) >> bit_positions) & 1).float()

        weights = (2 ** torch.arange(max_bits, device=abs_error.device)).float()

        weighted_error = (bit_errors * weights).sum(dim=-1)

        error_norm = weighted_error / weights.sum()

        lambda_risk = 20

        risk = torch.log(torch.exp(lambda_risk * error_norm).mean(dim=1)) / lambda_risk

        return (-10.0 * risk).clamp(min=-100.0)

    # =========================================================================
    # Interfaz MCTS
    # =========================================================================

    def rollout_from_state(self, state_dict, n_rollouts=None, rollout_depth=None):
        """
        Ejecuta N rollouts aleatorios en paralelo desde un estado del árbol MCTS.
        Todos los rollouts se evaluan con los MISMOS test cases exhaustivos
        (todos los pares A*B posibles), garantizando determinismo en la evaluacion.

        En modo incremental: los rewards se acumulan por columna durante el rollout.
        En modo original: reward único al final del episodio.

        Args:
            state_dict:    dict de BinaryMathEnv.get_state() — estado del nodo.
            n_rollouts:    cuántos rollouts paralelos (por defecto n_envs).
            rollout_depth: pasos máximos por rollout (por defecto CC).

        Returns:
            (n_rollouts,) tensor float32 — recompensas de cada simulación.
        """
        if n_rollouts is None:
            n_rollouts = self.n_envs
        n_rollouts = min(n_rollouts, self.n_envs)

        if rollout_depth is None:
            rollout_depth = self.CC

        env_idx = list(range(n_rollouts))
        self.reset(env_idx)
        self._load_state(state_dict, env_idx)

        # Rollouts aleatorios vectorizados
        for _ in range(rollout_depth):
            active = ~self.done[:n_rollouts]
            if not active.any():
                break
            actions = torch.randint(0, self.n_actions, (n_rollouts,),
                                    device=self.device)
            self._step_range(actions, n_rollouts)

        return self.rewards[:n_rollouts].clone()

    def _step_range(self, actions, n):
        """step() restringido a los primeros n entornos (evita copias)."""
        active   = ~self.done[:n]
        writable = active & (self.cursor_pos[:n] < self.CC)

        if not writable.any():
            return

        idx = writable.nonzero(as_tuple=True)[0]

        if self.incremental:
            flat_cells = self._cursor_to_flat(self.cursor_pos[idx])
            self.suma_grid[idx, flat_cells] = actions[idx].to(torch.int16)
        else:
            cols = self.cursor_pos[idx]
            self.suma_grid[idx, cols] = actions[idx].to(torch.int16)

        self.cursor_pos[idx] += 1

        if self.incremental:
            col_complete = writable & (self.cursor_pos[:n] % self.height == 0)
            if col_complete.any():
                comp_idx = col_complete.nonzero(as_tuple=True)[0]
                bp_vals  = (self.cursor_pos[comp_idx] // self.height - 1).long()

                for bp in bp_vals.unique():
                    # Construir máscara full-size para _evaluate_column_batch
                    full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                    full_mask[comp_idx[bp_vals == bp]] = True
                    self._evaluate_column_batch(int(bp.item()), full_mask)

            just_done = writable & (self.cursor_pos[:n] >= self.CC)
            if just_done.any():
                full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                full_mask[:n] = just_done
                self._penalize_overflow(full_mask)
                self.done[full_mask] = True

        else:
            just_done = writable & (self.cursor_pos[:n] >= self.CC)
            if just_done.any():
                # Construir máscara full-size para _evaluate_batch
                full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
                full_mask[:n] = just_done
                r = self._evaluate_batch(full_mask)
                self.rewards[full_mask] = r
                self.done[full_mask]    = True

    # =========================================================================
    # Logging por columna (modo incremental)
    # =========================================================================

    def enable_column_logging(self):
        """
        Activa buffers de logging por columna.
        Disponible solo en modo incremental.

        Buffers creados:
          _col_error_log: (n_envs, grid_size) — error medio por columna
          _carry_log:     (n_envs, grid_size+1) — carry medio por columna
        """
        if not self.incremental:
            raise RuntimeError("enable_column_logging() solo disponible con incremental=True")
        self._logging_enabled = True
        self._col_error_log = torch.zeros(self.n_envs, self.grid_size,
                                          dtype=torch.float32, device=self.device)
        self._carry_log = torch.zeros(self.n_envs, self.grid_size + 1,
                                      dtype=torch.int32, device=self.device)

    def reset_column_logs(self, env_indices=None):
        """Resetea los buffers de logging. Útil entre episodios."""
        if not self._logging_enabled:
            return
        if env_indices is None:
            self._col_error_log.fill_(0.0)
            self._carry_log.fill_(0)
        else:
            idx = torch.as_tensor(env_indices, device=self.device)
            self._col_error_log[idx] = 0.0
            self._carry_log[idx] = 0

    def get_column_stats(self):
        """
        Devuelve las métricas acumuladas por columna.

        Returns:
            dict con:
              'col_errors': (n_envs, grid_size) — error XOR medio por columna
              'carry_hist': (n_envs, grid_size+1) — carry medio por columna
        """
        if not self._logging_enabled:
            raise RuntimeError("Logging no activado. Llamar enable_column_logging() primero.")
        return {
            'col_errors': self._col_error_log.cpu(),
            'carry_hist': self._carry_log.cpu(),
        }

    def _log_column_stats(self, bit_pos, bit_error, carry_out, idx):
        """Interna: acumula métricas tras evaluar una columna."""
        self._col_error_log[idx, bit_pos] = bit_error.mean(dim=1)
        self._carry_log[idx, bit_pos + 1] = carry_out.float().mean(dim=1).to(torch.int32)

    # =========================================================================
    # Interoperabilidad con BinaryMathEnv (get_state / set_state)
    # =========================================================================

    def _load_state(self, state_dict, env_indices):
        """
        Carga un estado de BinaryMathEnv en un grupo de entornos CUDA.
        Convierte la suma_grid de strings a índices numéricos.

        En modo incremental: recalcula carry_in para columnas ya llenas,
        sin acumular reward (solo reconstruye el estado de carry).

        Args:
            state_dict:  dict devuelto por BinaryMathEnv.get_state().
            env_indices: lista de entornos destino.
        """
        grid_strs  = state_dict['suma_grid']
        cursor_val = state_dict['cursor_position']

        grid_indices = [self._action_to_idx.get(s, -1) for s in grid_strs]
        grid_tensor  = torch.tensor(grid_indices, dtype=torch.int16,
                                    device=self.device)

        idx = torch.as_tensor(env_indices, device=self.device)
        # Broadcast explícito para evitar indexing issues en CUDA
        grid_expanded = grid_tensor.unsqueeze(0).expand(len(idx), -1)
        self.suma_grid[idx]  = grid_expanded
        self.cursor_pos[idx] = cursor_val
        self.done[idx]       = False
        self.rewards[idx]    = 0.0

        if self.incremental:
            self.carry_in[idx] = 0
            if cursor_val > 0:
                self._recompute_carries_silent(env_indices)

    def get_single_state(self, env_idx):
        """
        Devuelve el estado de un entorno individual como diccionario
        compatible con BinaryMathEnv.set_state().

        Args:
            env_idx: índice del entorno.

        Returns:
            dict con 'suma_grid' (lista de strings) y 'cursor_position'.
        """
        grid = self.suma_grid[env_idx].cpu().tolist()
        grid_strs = [
            self.possible_actions[idx] if idx >= 0 else ' '
            for idx in grid
        ]
        return {
            'suma_grid':       grid_strs,
            'cursor_position': int(self.cursor_pos[env_idx].item()),
        }

    def clone_env(self, src_idx, dst_idx):
        """
        Clona un entorno dentro del batch (operación in-place, sin copias CPU).

        Args:
            src_idx: entorno origen.
            dst_idx: entorno(s) destino (int o lista).
        """
        self.suma_grid[dst_idx]  = self.suma_grid[src_idx]
        self.cursor_pos[dst_idx] = self.cursor_pos[src_idx]
        self.done[dst_idx]       = self.done[src_idx]
        self.rewards[dst_idx]    = self.rewards[src_idx]
        if self.incremental:
            self.carry_in[dst_idx] = self.carry_in[src_idx]

    # =========================================================================
    # Información / estadísticas
    # =========================================================================

    @property
    def active_count(self):
        """Número de entornos aún en ejecución."""
        return int((~self.done).sum().item())

    @property
    def completed_count(self):
        """Número de entornos que ya terminaron."""
        return int(self.done.sum().item())

    def summary(self):
        """Imprime estadísticas rápidas del batch actual."""
        done_rewards = self.rewards[self.done]
        print(f"Device:      {self.device}")
        print(f"Incremental: {self.incremental}")
        print(f"Entornos:    {self.n_envs}  |  activos: {self.active_count}  |  completados: {self.completed_count}")
        if done_rewards.numel() > 0:
            print(f"Reward       min={done_rewards.min():.4f}  "
                  f"mean={done_rewards.mean():.4f}  "
                  f"max={done_rewards.max():.4f}")
        else:
            print("Reward       sin datos aún")
