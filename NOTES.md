# MCTS-CUCA-CACA: Monte Carlo Tree Search para Multiplicadores Binarios

## Quick Start

```bash
# Versión principal con TensorBoard (recomendado)
python3 mcts_with_tensorboard.py --bits 2 --height 2 --iterations 2000 --c 2.0 --output ./mcts_results

# Versión rápida sin TensorBoard
python3 mcts_balanced_complete.py --bits 2 --height 2 --iterations 500 --output ./mcts_results

# Ver tablero de TensorBoard
tensorboard --logdir ./runs
```

Abre en navegador: **http://localhost:6006**

---

## Arquitectura del Proyecto

### Environment (`Environment/`)
- **env_cuda.py** — Principal. Simula tabla de productos parciales con evaluación CUDA
  - Columna-mayor (LSB-first)
  - Modo incremental: reward por columna completada
  - Paralelo: N entornos simultáneos en GPU
- **env_cuda_optimized.py** — Versión con OOM prevention
- **env_base.py** — Base con lógica común
- **env_adaptive.py** — Variante adaptativa
- **environment.py** — Wrapper compatible

### Algoritmos MCTS (`*.py` en raíz)

#### Producción
- **mcts_with_tensorboard.py** ⭐ **MAIN**
  - TensorBoard real-time logging
  - Coeficiente exploración ajustable (default c=2.0)
  - Árbol jerárquico equilibrado
  - Visualización de política final
  - Réplica: `mcts_balanced_complete.py` sin TensorBoard

#### Variantes UCT/PUCT
- **mcts_uct_v1.py** — Educativo con logs detallados
- **mcts_uct_v2_improvements.py** — UCT + Inverse UCB
- **mcts_uct_v3_rave.py** — UCT + RAVE (Rapid Action Value Estimation)
- **mcts_uct_v4_heuristic.py** — UCT + RAVE + Heurística (más completo)
- **mcts_puct_implementation.py** — PUCT para espacios grandes
- **mcts_logger.py** — Módulo TensorBoard compartido

#### Benchmarks
- **mcts_simple_benchmark.py** — Benchmark general con reporte HTML

### Tests
- `test_*.py` — 7 archivos de test del entorno y compatibilidad

### Utilidades
- `verilog/` — Simulador Verilog (referencia)
- `run_benchmark_hd.sh`, `run_all_benchmarks.sh` — Scripts de ejecución batch

---

## Parámetros Clave

### Exploración vs Explotación
```python
c = 1.41   # Balanceado (estándar)
c = 2.0    # Más exploración ✅ (recomendado para espacios grandes)
c = 3.0+   # Exploración agresiva
```

**Efecto:** Mayor `c` → árbol más ancho y profundo → mejor cobertura de espacio de acciones

### Configuración del Árbol
```python
min_children=2      # Hijos mínimos para descender en selección
min_visits=2        # Visitas mínimas para descender
incremental=True    # Reward por columna (vs. solo al final)
```

### Environment
```python
Bits=2              # Bits del multiplicador (2 = 18 acciones)
height=2            # Filas de tabla (2×2 → 8 celdas)
n_envs=1            # Entornos paralelos (GPU)
```

---

## TensorBoard Metrics

Accede a http://localhost:6006 después de ejecutar `mcts_with_tensorboard.py`:

### Pestaña "Scalars"
- **Reward/iteration** — Reward por iteración individual
- **Reward/best_so_far** — Mejor reward encontrado hasta ahora
- **Reward/avg_100** — Promedio móvil últimas 100 iteraciones
- **Tree/total_nodes** — Nodos totales en árbol
- **Tree/max_depth** — Profundidad máxima alcanzada
- **Tree/avg_depth** — Profundidad promedio
- **Tree/root_children** — Hijos del nodo raíz
- **Tree/branching_factor** — Factor de ramificación promedio
- **Tree/selection_depth** — Profundidad donde ocurre selección

### Interpretación
- **Convergencia lenta** → aumentar `c` (más exploración)
- **Árbol no crece** → revisar `min_children` o aumentar `iterations`
- **Branching factor > 2** → exploración equilibrada ✅
- **Max depth creciendo** → profundidad activa ✅

---

## Problema Resuelto: Árbol Estancado en Profundidad

### Síntoma
```
tree_size = 18 (constante)
depth = 1 (nunca crece)
```

### Causa
Condición `while node.children and node.fully_expanded()` nunca permitía descender con branching factor grande (18 acciones).

### Solución Aplicada
Reemplazar `fully_expanded()` con `can_descend(min_children=2)`:
```python
def can_descend(self, min_children=2, min_visits=2):
    return (len(self.children) >= min_children and
            self.visits >= min_visits and
            len(self.children) > 0)
```

**Cambios necesarios:**
1. Agregar método `can_descend()` a MCTSNode
2. Cambiar `while node.children and node.fully_expanded()` → `while node.can_descend(min_children, min_visits)`
3. Agregar `depth` tracking en nodos

### Resultado (Primera vez)
```
ANTES:  depth=1, nodes=19 ❌
DESPUÉS: depth=8, nodes=277 ✅
```

### Problema Recurrente: Árbol Estancado en 511 Nodos (Abril 2026)

**Síntoma:**
```
nodes=511 (constante durante 70k iteraciones)
depth=8 (constante)
bf=2.00 (branching factor muy bajo)
```

**Causa Identificada:**
Condición de selección `can_descend(min_children=2, min_visits=2)` era demasiado restrictiva:
- Requería 2 hijos visitados antes de descender
- Pero después de expandir, cada nodo nuevo solo tiene 1 hijo
- Esto crea cuello de botella: el nodo necesita 2 visitas Y 2 hijos antes de poder seguir

**Solución Aplicada (v2):**
Cambiar línea de selección en ambos archivos:
```python
# ANTES (restrictivo)
while node.can_descend(self.min_children, self.min_visits) and not done:

# DESPUÉS (permisivo)
while node.can_descend(min_children=1, min_visits=1) and not done:
```

Cambios:
1. `mcts_with_tensorboard.py` línea 143: `min_children=1, min_visits=1`
2. `mcts_balanced_complete.py` línea 118: `min_children=1, min_visits=1`

**Efecto esperado:**
- Árbol se expande más libremente en profundidad
- bf aumentará gradualmente conforme se exploran más ramas
- Mejor cobertura del espacio de búsqueda

---

## Solución Final (Abril 2026 - Descubrimiento Root Cause)

**PROBLEMA FUNDAMENTAL IDENTIFICADO:**

Tanto `mcts_correct_design.py` como `mcts_with_tensorboard.py` no seguían la **lógica correcta de MCTS**. El criterio de descenso (`can_descend()`) permitía que un nodo con solo 1 hijo descendiera indefinidamente, **evitando exploración lateral**.

**La versión `v1.py` y `v4.py` usaban el patrón correcto:**

```python
def is_fully_expanded(self):
    """Solo True si NO hay untried_actions"""
    return len(self.untried_actions) == 0

# En selección:
while node.is_fully_expanded() and node.children:  # ← MCTS CORRECTO
    node = node.best_child(c)
```

**Por qué funciona:**
- Si un nodo tiene `untried_actions`, el while **sale inmediatamente**
- No puede descender hasta haber expandido TODAS sus acciones
- Esto **fuerza exploración lateral** antes de permitir profundidad

**Cambios Aplicados a `mcts_correct_design.py`:**

1. Reemplazar `can_descend(min_children, min_visits)` por `is_fully_expanded()`
2. Cambiar línea ~170: `while node.is_fully_expanded() and node.children and not done:`

**Resultados ANTES del fix:**
```
Iteraciones: 500
Nodos: 9
Depth: 8
BF: 1.00 (árbol lineal)
```

**Resultados DESPUÉS del fix:**
```
Iteraciones: 500
Nodos: 501 (≈1 por iteración) ✅
Depth: 4 ✅
BF: 6.58 (exploración amplia) ✅
Best reward: -1.9480 ✅
```

**Lección:** MCTS canónico requiere `is_fully_expanded()`, NO restricciones sobre min_children/min_visits.

### Error Crítico Descubierto: Desincronización de Estado (Abril 2026)

**Síntoma:**
```
Árbol binario perfecto: 511 nodos (2^9-1)
Profundidad exacta: 8
Branching factor exacto: 2.00
Solo 500 nodos creados en 100,000 iteraciones
```

**Causa Raíz:**
El MCTS **guarda el estado en cada nodo pero NUNCA lo restaura al descender**:

```python
# INICIO DE ITERACIÓN - Solo carga raíz
self.env.reset([0])
self.env._load_state(self.root.state, [0])  # ✅ Raíz OK

# SELECCIÓN - Desciende pero NO restaura estado
while node.can_descend(...):
    best_child = node.best_child(c=self.c)
    reward, done_batch = self.env.step(action)  # ❌ Environment en estado INCORRECTO
    node = best_child  # ❌ best_child.state NO es el estado actual del env
```

**Consecuencias:**
1. El environment no está en el estado que corresponde a `node`
2. Rewards/done señales son incorrectos
3. UCT calcula valores basados en datos inconsistentes
4. Selección/Expansión favorece ramas que parecen mejores pero están malcalculadas
5. La mayoría de iteraciones NO expanden (porque state es inválido)

**Solución Correcta:**
Reproducir **secuencia de acciones desde raíz** en lugar de guardar/restaurar estado:

```python
def reproduce_state(self, node):
    """Reproduce el estado del environment para llegar a un nodo"""
    self.env.reset([0])
    # Obtener secuencia de acciones desde raíz hasta node
    for action_idx in node.get_action_sequence():
        self.env.step(action_idx)

# En selection:
while node.can_descend(...):
    best_child = node.best_child(c=self.c)
    self.reproduce_state(best_child)  # ✅ RESTAURA ESTADO CORRECTAMENTE
    reward, done_batch = self.env.step(action)
    node = best_child
```

**Implementación Corregida:**
- `mcts_correct_design.py` ⭐ NUEVA VERSIÓN CORRECTA
  - Reproduce acciones desde raíz en cada selección
  - No guarda estado incompleto
  - Garantiza consistencia environment ↔ árbol

**Prueba:**
```bash
python3 mcts_correct_design.py --bits 2 --height 2 --iterations 100000 --c 2.0
```

Expected resultado:
- nodes >> 511 (probablemente 10,000+)
- depth > 8 (debería crecer)
- Mejor convergencia de rewards

---

## Implementación Incremental del Environment

El `incremental=True` en BinaryMathEnvCUDA permite:

1. **Llenado columna-mayor (LSB → MSB)**
   - Cursor mueve por columnas completadas
   
2. **Reward por columna completada**
   - Al terminar cada columna se computa XOR ponderado
   - Range: [-10, 0] (0 = circuito perfecto)
   - Permitedetectar mejoras parciales SIN esperar fin de episodio

3. **Ventaja para MCTS**
   - Feedback más frecuente
   - Mejor gradiente de mejora
   - Convergencia más rápida

---

## Guía UCT (Upper Confidence bounds applied to Trees)

### Fórmula UCT
```
UCT = exploitation + exploration
    = Q(s) / N(s) + c * sqrt(ln(N_parent) / N(s))
```

Donde:
- `Q(s)` = suma acumulada de rewards
- `N(s)` = número de visitas
- `c` = coeficiente exploración

### Versiones Implementadas

| Versión | Descripción | Uso |
|---------|-------------|-----|
| v1 | Base con logs | Educación |
| v2 | + Inverse UCB | Benchmark |
| v3 | + RAVE | Mejora rápida |
| v4 | + Heurística | Producción |

### RAVE (Rapid Action Value Estimation)
Combina valores de todas las posiciones donde aparece una acción, no solo la rama actual.
Mejora significativa en primeras iteraciones.

---

## Guía PUCT (Polynomial Upper Confidence bounds applied to Trees)

Evolución de UCT para espacios muy grandes (miles de acciones):

```
PUCT = Q(s) + c * P(s) * sqrt(N_parent) / (1 + N(s))
```

Donde:
- `P(s)` = prior probability (política inicial)
- Exponente más fuerte en `sqrt(N_parent)`

**Ventaja:** Mejor exploración en espacios grandes
**Trade-off:** Requiere estimación de priors

---

## Estructura de Datos de MCTSNode

```python
class MCTSNode:
    state          # Estado del entorno en este nodo
    parent         # Referencia al nodo padre
    action         # Acción que llevó aquí
    depth          # Profundidad en el árbol
    children       # Dict: action_idx → MCTSNode
    visits         # Contador de visitas
    reward_sum     # Suma acumulada de rewards
    untried_actions # Lista de acciones sin probar
```

---

## Ejecutar Tests

```bash
# Entorno básico
python3 test_incremental_env.py

# Multiplicador perfecto
python3 test_perfect_multiplier.py

# Integración MCTS + Entorno
python3 test_mcts_with_new_env.py

# Clonación de estado
python3 test_cloning.py

# GPU optimizada
python3 test_cuda_optimized.py

# Feedback de reward
python3 test_reward_feedback.py

# Multiplicador 2x2
python3 test_perfect_multiplier_height2.py
```

---

## Troubleshooting

### "Tree size stuck at 1 o no crece"
→ Ver `can_descend()` o aumentar `iterations`

### "Rewards no convergen"
→ Aumentar `c` (más exploración) o `iterations`

### "TensorBoard no muestra datos"
→ Verificar: `ls runs/` debe tener directorios recientes
→ Reiniciar tensorboard después de ejecución

### "CUDA out of memory"
→ Reducir `n_envs` o usar `env_cuda_optimized.py`

### "Profundidad muy baja"
→ Aumentar `min_children` permite descender más rápido

---

## Referencias

- **UCT**: Kocsis & Szepesvári (2006)
- **RAVE**: Gelly & Silver (2007)
- **PUCT**: AlphaGo paper (Silver et al., 2016)

---

## Estructura Esperada Post-Limpieza

```
MCTS_CUCA_CACA/
├── Environment/
│   ├── env_cuda.py              (principal)
│   ├── env_cuda_optimized.py
│   ├── env_base.py
│   ├── env_adaptive.py
│   ├── environment.py
│   └── __init__.py
├── mcts_with_tensorboard.py     ⭐ MAIN
├── mcts_balanced_complete.py    (alternativa rápida)
├── mcts_uct_v1.py               (educativo)
├── mcts_uct_v2_improvements.py
├── mcts_uct_v3_rave.py
├── mcts_uct_v4_heuristic.py     (más completo)
├── mcts_puct_implementation.py   (espacios grandes)
├── mcts_logger.py               (módulo compartido)
├── mcts_simple_benchmark.py
├── test_cloning.py
├── test_incremental_env.py
├── test_perfect_multiplier.py
├── test_perfect_multiplier_height2.py
├── test_reward_feedback.py
├── test_mcts_with_new_env.py
├── test_cuda_optimized.py
├── verilog/
├── run_benchmark_hd.sh
├── run_all_benchmarks.sh
└── NOTES.md                     ← este archivo
```

---

**Last updated:** 2026-04-14  
**Status:** Workspace limpio, listo para desarrollo activo
