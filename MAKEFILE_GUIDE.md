# 🚀 Guía de Makefile - AlphaZero Producer-Consumer

Sistema automatizado para entrenar AlphaZero de forma distribuida con retroalimentación continua.

---

## 📋 Paso 1: Configurar Variables

Edita el Makefile directamente o crea un archivo `.makefile.local`:

```bash
# Opción 1: Editar directamente el Makefile (líneas 5-28)
nano Makefile

# Cambiar estas líneas:
USER_A = tu_usuario_a
HOST_A = ip_o_host_a
USER_B = tu_usuario_b
HOST_B = ip_o_host_b
```

**Ejemplo de valores reales:**
```makefile
USER_A = servergmun
HOST_A = 192.168.1.10
USER_B = otro_usuario
HOST_B = 192.168.1.20
```

O crea un archivo local (que NO se comiteará):

```bash
cat > Makefile.local << 'EOF'
USER_A = servergmun
HOST_A = localhost
USER_B = servergmun
HOST_B = localhost
BITS = 4
N_SIM_PRODUCER = 100
TRAIN_STEPS = 60
EOF

make -f Makefile.local help  # Usa así
```

---

## ✅ Paso 2: Validar Configuración

```bash
# Verifica que todo está bien configurado
make validate

# Output esperado:
# [*] Validando configuración...
# [+] Conexión OK
# [+] Proyecto encontrado
# [+] ¡Validación completa!
```

Si falla, verifica:
- ¿SSH funciona? `ssh user_a@host_a "echo OK"`
- ¿Ruta proyecto? `ssh user_a@host_a "ls ~/MCTS_CUCA_CACA/alphazero/main.py"`
- ¿Credenciales? Configura clave SSH sin contraseña

---

## 🛠️ Paso 3: Setup Inicial

```bash
# Prepara ambos workstations
make setup

# Esto:
# ✓ Crea carpetas en A: games_prod, checkpoints, logs
# ✓ Crea carpetas en B: games_all, checkpoints, logs
# ✓ Sincroniza trainer.py mejorado
```

---

## 🎯 Paso 4: Iniciar Sistema

### Terminal 1: Inicia Producer (WORKSTATION A)

```bash
make start-producer

# Output:
# [*] Iniciando PRODUCER en user_a@host_a...
# [+] Producer iniciado. Ver logs: make monitor-producer
```

### Terminal 2: Monitorea Producer

```bash
make monitor-producer

# Verás logs como:
# [2026-04-24 10:30:45] Generando juegos con MCTS...
# [2026-04-24 10:31:12] Juego 1/20 completado
# [2026-04-24 10:31:45] Guardando 160 muestras
```

### Terminal 3: Inicia Consumer (WORKSTATION B)

```bash
make start-consumer

# Output:
# [*] Iniciando CONSUMER en user_b@host_b...
# [+] Consumer iniciado. Ver logs: make monitor-consumer
```

### Terminal 4: Monitorea Consumer

```bash
make monitor-consumer

# Verás logs como:
# [2026-04-24 10:32:00] Sincronizando juegos...
# [2026-04-24 10:32:15] Total juegos: 520
# [2026-04-24 10:32:30] Entrenando red neuronal...
# [2026-04-24 10:45:00] Entrenamiento completado
```

---

## 📊 Paso 5: Monitorear Progreso

### Ver Estado Actual

```bash
make status

# Output:
# === ESTADO PRODUCER ===
# user_a    1234  producer.sh
# user_a    1235  python3 alphazero/main.py
#
# === ESTADO CONSUMER ===
# user_b    5678  consumer.sh
# user_b    5679  python3 alphazero/main.py
```

### Ver Métricas de Mejora

```bash
make metrics

# Output:
# === MÉTRICAS DE MEJORA ===
#
# JUEGOS GENERADOS (Producer):
#   Total: 4520
#
# JUEGOS DESCARGADOS (Consumer):
#   Total: 4380
#
# CHECKPOINTS GENERADOS:
#   checkpoint_00050.pt - 4.2M bytes
#   checkpoint_00040.pt - 4.2M bytes
#
# TAMAÑO TOTAL (Producer):
#   320M
#
# TAMAÑO TOTAL (Consumer):
#   305M
```

### Visualizar Buffer (Datos de Entrenamiento)

```bash
make inspect

# Genera PNG con 8 muestras aleatorias del buffer
# Muestra evolución de topologías
```

---

## 🔄 Ciclo de Mejora Automático

Cuando todo está corriendo:

```
T=0h    Producer (red v0) → 100 juegos → Consumer entrena → Red v1
T=30m   Producer (red v1) → 100 juegos → Consumer entrena → Red v2
T=60m   Producer (red v2) → 100 juegos → Consumer entrena → Red v3
T=90m   Producer (red v3) → 100 juegos → Consumer entrena → Red v4
...

Las topologías generadas cada ciclo son MEJORES porque la red es mejor.
```

---

## 🎛️ Ajustar Parámetros

### Para Entrenar Más Rápido

```makefile
N_SIM_PRODUCER = 80          # Menos simulaciones MCTS
ITERATIONS_PRODUCER = 30     # Menos iteraciones por ciclo
TRAIN_STEPS = 40             # Menos pasos de gradient
CONSUMER_SLEEP = 900         # Ciclos más cortos (15 min)
```

### Para Mejor Calidad

```makefile
N_SIM_PRODUCER = 150         # Más simulaciones MCTS
ITERATIONS_PRODUCER = 100    # Más iteraciones
TRAIN_STEPS = 100            # Más pasos de gradient
BATCH_SIZE = 512             # Batch más grande
```

### Para Hardware Limitado (GPU 8GB)

```makefile
N_SIM_PRODUCER = 60
GAMES_PER_ITER = 10
TRAIN_STEPS = 20
BATCH_SIZE = 128
```

---

## 🛑 Parar Sistema

```bash
# Parar solo producer
make stop-producer

# Parar solo consumer
make stop-consumer

# Parar ambos y limpiar logs
make cleanup
```

---

## 🔍 Troubleshooting

### Logs no aparecen

```bash
# Ver logs manualmente
ssh user_a@host_a "tail -50 /tmp/producer.log"
ssh user_b@host_b "tail -50 /tmp/consumer.log"
```

### Error: "No se pudo conectar"

```bash
# Verifica SSH
ssh user_a@host_a "echo OK"

# Si falla, configura clave sin contraseña
ssh-copy-id -i ~/.ssh/id_rsa.pub user_a@host_a
```

### Consumer no descarga juegos

```bash
# Verifica que producer está generando juegos
ssh user_a@host_a "ls -l ~/MCTS_CUCA_CACA/games_prod/ | tail -5"

# Verifica manualmente rsync
rsync -avz user_a@host_a:~/MCTS_CUCA_CACA/games_prod/ /tmp/test/
```

### GPU Memory Error

```makefile
# Reduce BATCH_SIZE y N_SIM
BATCH_SIZE = 128
N_SIM_PRODUCER = 60
```

---

## 📈 Validar Mejora

Cada 1-2 horas, ejecuta:

```bash
# Ver cuántos juegos se han generado
make metrics

# Comparar con hora anterior
# Esperado: +200 juegos/hora en producer
#           +1 checkpoint nuevo en consumer

# Generar snapshot visual
make inspect

# Comparar snapshots:
# Temprano: topologías aleatorias/malas
# Medio:    alguna estructura buena
# Tardío:   topologías sofisticadas
```

---

## 🎯 Ejemplo Completo (5 minutos)

```bash
# Terminal 1: Setup
make validate
make setup

# Terminal 2: Producer
make start-producer
sleep 2
make monitor-producer

# Terminal 3 (en otra máquina o en background)
make start-consumer
make monitor-consumer

# Terminal 4: Monitorear progreso cada 10 min
watch -n 600 'make metrics'

# O manual:
make metrics      # Ver números
make inspect      # Ver imágenes
make status       # Ver procesos vivos
```

---

## 🚀 Parámetros Recomendados por GPU

### GTX 1070 Ti (8GB)
```makefile
N_SIM_PRODUCER = 120
GAMES_PER_ITER = 12
TRAIN_STEPS = 40
BATCH_SIZE = 256
ITERATIONS_PRODUCER = 500
ITERATIONS_CONSUMER = 100
```

### RTX 3080 (10GB)
```makefile
N_SIM_PRODUCER = 150
GAMES_PER_ITER = 16
TRAIN_STEPS = 60
BATCH_SIZE = 512
ITERATIONS_PRODUCER = 500
ITERATIONS_CONSUMER = 200
```

### RTX A100 (40GB)
```makefile
N_SIM_PRODUCER = 200
GAMES_PER_ITER = 32
TRAIN_STEPS = 100
BATCH_SIZE = 1024
ITERATIONS_PRODUCER = 1000
ITERATIONS_CONSUMER = 300
```

---

## 📝 Resumen de Commands

```bash
make help              # Ver todos los targets
make validate          # Verificar configuración
make setup             # Preparar workstations
make start-producer    # Iniciar generador
make start-consumer    # Iniciar entrenador
make stop-producer     # Parar generador
make stop-consumer     # Parar entrenador
make status            # Ver procesos activos
make monitor-producer  # Ver logs producer
make monitor-consumer  # Ver logs consumer
make metrics           # Ver estadísticas
make inspect           # Generar visualización
make cleanup           # Limpiar todo
```

---

¡Listo! El sistema está completamente automatizado. 🎉

