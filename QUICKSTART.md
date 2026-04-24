# ⚡ Quick Start - AlphaZero Producer-Consumer

**5 pasos para empezar en 10 minutos**

---

## 📋 Paso 1: Valida tu setup

```bash
cd ~/MCTS_CUCA_CACA
bash validate_setup.sh
```

Debería ver `✓ VALIDACIÓN COMPLETA` al final.

---

## 🎛️ Paso 2: Configura tus máquinas en el Makefile

```bash
nano Makefile

# Cambia estas líneas (busca la sección CONFIGURACIÓN PRINCIPAL):
USER_A = tu_usuario_en_workstation_a
HOST_A = ip_o_hostname_a
USER_B = tu_usuario_en_workstation_b  
HOST_B = ip_o_hostname_b
```

**Ejemplo:**
```makefile
USER_A = servergmun
HOST_A = 192.168.1.100
USER_B = juan
HOST_B = 192.168.1.101
```

---

## ✅ Paso 3: Valida configuración SSH

```bash
make validate

# Deberías ver:
# [+] Conexión OK (A)
# [+] Conexión OK (B)
# [+] ¡Validación completa!
```

Si falla, lee `MAKEFILE_GUIDE.md` sección "Troubleshooting".

---

## 🛠️ Paso 4: Setup inicial (una sola vez)

```bash
make setup

# Crea carpetas en ambos workstations
```

---

## 🚀 Paso 5: Inicia el sistema

### Terminal 1 (Producer)
```bash
make start-producer
```

### Terminal 2 (Consumer)
```bash
make start-consumer
```

### Terminal 3 (Monitoreo)
```bash
# Ver logs en tiempo real
make monitor-producer

# O en otra terminal:
make monitor-consumer

# O ver métricas:
watch -n 60 'make metrics'
```

---

## 📊 Monitorear Mejora

Cada 1 hora, ejecuta:

```bash
make metrics

# Verás:
# JUEGOS GENERADOS: debería crecer ~200-400 por hora
# CHECKPOINTS: nuevo cada 30 minutos
```

---

## 🛑 Parar Sistema

```bash
make cleanup  # Para todo y limpia logs
```

---

## 🎯 Esperado después de 1 hora

```
Juegos generados: 300-500
Checkpoints: 2-3 nuevos
GPU A: 90-95% utilización MCTS
GPU B: 90-95% utilización training
```

---

## 📚 Documentación Completa

- `MAKEFILE_GUIDE.md` - Guía detallada de todos los comandos
- `README.md` - Información del proyecto
- `PERSISTENCE_GUIDE.md` - Cómo continuar entrenamientos

---

## 🆘 Problemas?

```bash
# Ver log completo del producer
ssh USER_A@HOST_A "tail -100 /tmp/producer.log"

# Ver log completo del consumer
ssh USER_B@HOST_B "tail -100 /tmp/consumer.log"

# Matar procesos manualmente si es necesario
ssh USER_A@HOST_A "pkill -9 python3"
```

---

**¡Listo! El sistema debería estar corriendo. 🎉**
