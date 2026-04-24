# ⚡ Quick Start - Sistema Distribuido Mejorado

**Cada workstation se controla a sí misma. Tú solo observas.**

---

## 🎯 Arquitectura Mejorada

```
Tu Máquina (Controlador Local)
├── Makefile.local (controla ambos remotamente)
│
├─── SSH ──→ WORKSTATION A (Producer)
│           ├── Makefile.producer (autónomo)
│           └── Genera juegos continuamente
│
└─── SSH ──→ WORKSTATION B (Consumer)
            ├── Makefile.consumer (autónomo)
            └── Entrena continuamente
```

Cada workstation:
- ✅ Se ejecuta **sin intervención manual**
- ✅ Descarga sus propios checkpoints/juegos
- ✅ Puede controlarse localmente o remotamente

---

## 📋 Paso 1: Edita configuración en Makefile.local

```bash
nano Makefile.local

# Cambia estas líneas:
USER_A = tu_usuario_a
HOST_A = ip_workstation_a
USER_B = tu_usuario_b
HOST_B = ip_workstation_b
```

---

## 📋 Paso 2: Edita configuración en Makefile.producer

```bash
nano Makefile.producer

# Líneas 9-14 (opcionales, por defecto están bien):
BITS ?= 4
N_SIM ?= 100
GAMES_PER_ITER ?= 20
ITERATIONS ?= 50

# Y setea los detalles del consumer (líneas 16-18):
CONSUMER_USER = tu_usuario_b
CONSUMER_HOST = ip_workstation_b
```

---

## 📋 Paso 3: Edita configuración en Makefile.consumer

```bash
nano Makefile.consumer

# Líneas 9-14 (opcional):
BITS ?= 4
TRAIN_STEPS ?= 60
BATCH_SIZE ?= 256
ITERATIONS ?= 100

# Y setea detalles del producer (líneas 18-20):
PRODUCER_USER = tu_usuario_a
PRODUCER_HOST = ip_workstation_a
```

---

## 🚀 Paso 4: Despliega los Makefiles en ambas máquinas

```bash
# Desde tu máquina local
cd ~/MCTS_CUCA_CACA
make -f Makefile.local deploy
```

Esto copia `Makefile.producer` y `Makefile.consumer` a cada workstation.

---

## 🎬 Paso 5: Inicia Producer (WORKSTATION A)

Abre **Terminal 1**:

```bash
ssh gmunserver@192.168.50.101
cd ~/MCTS_CUCA_CACA
make -f Makefile.producer start
```

Verás:
```
[*] Iniciando Producer...
[+] Producer iniciado!
Ver logs: tail -f /tmp/producer.log
```

Mantén esta sesión abierta o puedes desconectarte (el proceso sigue corriendo en nohup).

---

## 🎬 Paso 6: Inicia Consumer (WORKSTATION B)

Abre **Terminal 2**:

```bash
ssh gmunserver@192.168.50.102
cd ~/MCTS_CUCA_CACA
make -f Makefile.consumer start
```

Verás lo mismo. El consumer está ahora corriendo autónomamente.

---

## 📊 Paso 7: Monitorea desde tu máquina local

Abre **Terminal 3** (en tu máquina):

```bash
cd ~/MCTS_CUCA_CACA

# Ver estado de ambos
make -f Makefile.local status

# Ver métricas en tiempo real
watch -n 60 'make -f Makefile.local metrics'

# O monitorear logs de producer
make -f Makefile.local monitor-producer

# O logs de consumer (en otra terminal)
make -f Makefile.local monitor-consumer
```

---

## 🔄 Ciclo Automático

Una vez iniciados, **los procesos se ejecutan solos**:

```
T=0min:    Producer genera 100 juegos → Consumer descarga
T=5min:    Consumer entrena con 100 juegos → Red v1
T=35min:   Producer obtiene Red v1 → genera 100 juegos mejores
T=40min:   Consumer entrena con 200 juegos → Red v2
T=70min:   Producer obtiene Red v2 → genera 100 juegos mejores aún
...
```

**Las topologías mejoran automáticamente cada ciclo.**

---

## 🎛️ Controlar desde Local

Desde tu máquina, puedes:

```bash
# Ver estado actual de ambos
make -f Makefile.local status

# Ver métricas
make -f Makefile.local metrics

# Generar visualización del buffer
make -f Makefile.local inspect

# Parar producer
ssh gmunserver@192.168.50.101 "cd ~/MCTS_CUCA_CACA && make -f Makefile.producer stop"

# Parar consumer
ssh gmunserver@192.168.50.102 "cd ~/MCTS_CUCA_CACA && make -f Makefile.consumer stop"

# Parar ambos desde local (comando corto)
make -f Makefile.local stop

# Parar y limpiar todo
make -f Makefile.local cleanup
```

---

## 📈 Validar Mejora

Cada hora, ejecuta:

```bash
make -f Makefile.local metrics

# Deberías ver:
# PRODUCER:
#   Juegos generados: 800+ (y creciendo)
#   GPU: 90-95% utilización
#
# CONSUMER:
#   Juegos descargados: 700+
#   Checkpoints: 2-3 nuevos
#   GPU: 90-95% utilización
```

---

## 🎯 Ventajas del Sistema Mejorado

✅ **Autónomo:** Cada máquina se ejecuta sin intervención
✅ **Resiliente:** Si una terminal se desconecta, el proceso sigue corriendo
✅ **Escalable:** Fácil agregar más workstations
✅ **Distribuido:** No necesitas máquina "maestra"
✅ **Feedback Loop:** Red → Juegos → Red mejorada (automático)

---

## 🆘 Troubleshooting

### Ver logs localmente desde producer

```bash
ssh gmunserver@192.168.50.101 "tail -f /tmp/producer.log"
```

### Ver logs localmente desde consumer

```bash
ssh gmunserver@192.168.50.102 "tail -f /tmp/consumer.log"
```

### Verificar que procesos están corriendo

```bash
ssh gmunserver@192.168.50.101 "ps aux | grep -E 'producer|python3'"
ssh gmunserver@192.168.50.102 "ps aux | grep -E 'consumer|python3'"
```

### Matar procesos manualmente

```bash
# En producer
ssh gmunserver@192.168.50.101 "pkill -f producer.sh; pkill -f alphazero"

# En consumer
ssh gmunserver@192.168.50.102 "pkill -f consumer.sh; pkill -f alphazero"
```

---

## 📝 Resumen de Archivos

```
Tu máquina:
├── Makefile.local         ← Controla ambos remotamente
│
WORKSTATION A:
├── Makefile.producer      ← Se ejecuta autónomamente
├── games_prod/            ← Juegos generados
└── checkpoints_producer/  ← Checkpoints descargados

WORKSTATION B:
├── Makefile.consumer      ← Se ejecuta autónomamente
├── games_all/             ← Juegos descargados
└── alphazero/checkpoints/ ← Checkpoints generados
```

---

**¡Sistema listo para entrenar distribuido! 🚀**

