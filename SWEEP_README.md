# Barrido de Parámetros (Alpha, C_pw, c)

Hay dos scripts para hacer barridos de parámetros:

## 1. `sweep_params.py` - Barrido Predefinido Inteligente

Script listo para usar que ejecuta un barrido cuidadosamente diseñado según la lógica alpha-c-profundidad.

**Uso:**
```bash
cd /home/servergmun/MCTS_CUCA_CACA
python3 sweep_params.py
```

**Qué hace:**
- Define 9 configuraciones estratégicas agrupadas en 4 "regímenes"
- Régimen 1 (alpha bajo): maximiza profundidad
- Régimen 2-3: balance profundidad-amplitud
- Régimen 4: amplitud máxima (baseline actual)
- Ejecuta todos en secuencia
- Genera reporte CSV y tabla resumen

**Salida esperada:**
```
RESUMEN DEL BARRIDO
========================================================
Alpha    C_pw    c        Reward      Depth   Nodes    BF
0.25     1.50    1.80     -0.1234     12      2500     3.5
0.30     1.50    2.00     -0.2145     10      2300     3.2
...
0.50     2.00    2.50     -1.2301     4       10001    5.0

TOP 3 (por profundidad, luego reward)
========================================================
1. alpha=0.25, C_pw=1.5, c=1.8 → d=12, r=-0.1234, nodes=2500
2. alpha=0.30, C_pw=1.5, c=2.0 → d=10, r=-0.2145, nodes=2300
3. alpha=0.20, C_pw=1.0, c=1.5 → d=9, r=-0.3456, nodes=1800
```

---

## 2. `sweep_custom.py` - Barrido Personalizado Flexible

Script más flexible para explorar espacios de parámetros personalizados.

### Opción A: Usar presets
```bash
# Preset 1: Máxima profundidad
python3 sweep_custom.py --preset deep

# Preset 2: Balance
python3 sweep_custom.py --preset balanced

# Preset 3: Máxima amplitud
python3 sweep_custom.py --preset broad
```

### Opción B: Especificar rangos personalizados
```bash
# Barrido fino de alpha entre 0.15-0.35, C_pw entre 1.0-2.0, c entre 1.5-2.2
python3 sweep_custom.py \
  --alpha-range 0.15 0.35 0.05 \
  --C-pw-range 1.0 2.0 0.25 \
  --c-range 1.5 2.2 0.15
```

### Opciones generales
```bash
--bits INT              # Número de bits (default: 4)
--iterations INT        # Iteraciones MCTS (default: 10000)
--n-rollouts INT        # Rollouts paralelos (default: 32)
--preset STRING         # deep | balanced | broad
--alpha-range MIN MAX STEP
--C-pw-range MIN MAX STEP
--c-range MIN MAX STEP
```

---

## Interpretación de Resultados

Después de cada barrido, busca en los resultados:

### 1. Profundidad máxima
```python
# En la tabla resumen:
# Identifica qué configuración tiene el mayor final_depth
# Ese es el "ganador" en términos de capacidad de exploración
```

**Interpretación:**
- depth=4 (baseline): no llega a llenar el circuito (CC=32)
- depth=8-12: cubre ~25% de las celdas
- depth=16+: cobertura media-buena
- depth=32: circuito completo (objetivo)

### 2. Reward en función de profundidad
```
Mejor reward generalmente correlaciona con profundidad:
- depth=4:  reward ≈ -1.0 a -10 (malo)
- depth=8:  reward ≈ -0.5 a -2 (mejor)
- depth=16: reward ≈ -0.1 a -0.5 (bueno)
- depth=32: reward ≈ 0 (óptimo teórico)
```

### 3. Búsqueda de óptimo local

Usa esta fórmula simple para saber si seguir explorando:
```
optimalidad = final_depth / CC
- < 0.3:  lejos del óptimo, sigue probando valores menores de alpha
- 0.3-0.7: en la zona de búsqueda, refina alrededor de estos valores
- > 0.7:  acercándose, profundidad casi máxima
```

---

## Experimentos Recomendados

### Fase 1: Exploración amplia (rápido)
```bash
python3 sweep_custom.py --preset deep --iterations 5000
```
Esto te da una idea rápida de qué valores de alpha y C_pw son prometedores.

### Fase 2: Refinamiento (mediano)
```bash
# Basándote en los resultados de Fase 1, refina alrededor del mejor
python3 sweep_custom.py \
  --alpha-range 0.20 0.35 0.03 \
  --C-pw-range 1.0 1.7 0.1 \
  --c-range 1.5 2.0 0.1
```

### Fase 3: Validación (largo)
```bash
# Ejecuta los 3-5 mejores con 30000 iteraciones
python3 sweep_custom.py \
  --alpha-range 0.22 0.26 0.01 \
  --C-pw-range 1.2 1.5 0.1 \
  --c-range 1.6 1.9 0.1 \
  --iterations 30000
```

---

## Flujo Esperado Completo

1. **Ejecuta barrido inicial:**
   ```bash
   python3 sweep_params.py
   ```

2. **Analiza top 3:**
   ```
   Observa qué configuración tiene la mayor profundidad.
   ```

3. **Refina alrededor del ganador:**
   ```bash
   # Si el ganador fue alpha=0.25, refina:
   python3 sweep_custom.py \
     --alpha-range 0.20 0.30 0.02 \
     --C-pw-range 1.2 1.8 0.2 \
     --c-range 1.5 2.0 0.1
   ```

4. **Valida con iteraciones mayores:**
   ```bash
   # Los 2-3 mejores con 20000-30000 iteraciones
   python3 mcts_scalable.py --bits 4 --iterations 30000 \
     --alpha 0.25 --C-pw 1.5 --c 1.8 --output final_validation
   ```

---

## Notas Técnicas

### Estimación de tiempo
- **5000 iteraciones:** ~30s por config
- **10000 iteraciones:** ~60s por config
- **30000 iteraciones:** ~180s por config

**Para barrido de 9 configs × 10000 iter:** ~9-10 minutos total

### Almacenamiento
Cada ejecución crea un directorio `sweep_a*_cp*_c*.../` con:
- `stats.json` (métricas)
- `mcts_analysis.png` (gráficos)
- `events/` (logs de TensorBoard)
- `policy_visualization.png` (política final)

Estos pueden ocupar ~50-100MB por ejecución. Limpia con:
```bash
rm -rf sweep_a* sweep_*_results.csv
```

### Relación teórica (referencia)

Para recordar por qué estos parámetros importan:

```
alpha = 0.5  (actual)  → max_children ≈ 2 * sqrt(visits)
             → Con 10K iteraciones: depth máx ≈ 4

alpha = 0.25 (propuesto) → max_children ≈ 1.5 * (visits ^ 0.25)
             → Con 10K iteraciones: depth máx ≈ 12-16

alpha = 0.15 (agresivo) → max_children ≈ 1.0 * (visits ^ 0.15)
             → Con 10K iteraciones: depth máx ≈ 20+
```

---

## Troubleshooting

### Error: "No module named 'Environment'"
```bash
# Asegúrate de estar en el directorio correcto:
cd /home/servergmun/MCTS_CUCA_CACA
python3 sweep_params.py
```

### Memoria insuficiente (OOM)
```bash
# Reduce n_rollouts:
python3 sweep_custom.py --preset deep --n-rollouts 16

# O reduce iteraciones:
python3 sweep_custom.py --preset deep --iterations 5000
```

### Script tarda demasiado
```bash
# Empieza con Fase 1 (pocas configs, pocas iteraciones):
python3 sweep_custom.py --preset deep --iterations 5000
```

---

## Hipótesis a Validar

Basándome en el análisis teórico, espero que el barrido muestre:

✅ **Hipótesis 1:** alpha=0.25-0.35 es mejor que alpha=0.5 para profundidad
   - Métrica: final_depth significativamente mayor

✅ **Hipótesis 2:** C_pw=1.0-1.5 funciona mejor con alpha bajo
   - Métrica: reward mejora cuando C_pw se reduce con alpha

✅ **Hipótesis 3:** c=1.5-2.0 es mejor que c=2.5 para 4 bits
   - Métrica: c más bajo es mejor con alpha bajo

Si alguna hipótesis NO se valida, significa que hay dinámicas no capturadas en el modelo teórico → interesante para analizar más.

---

**¿Listos para ejecutar?** 🚀

Recomendación: Empieza con `sweep_params.py` (9 configs, 10 min) y luego refina basándote en los resultados.
