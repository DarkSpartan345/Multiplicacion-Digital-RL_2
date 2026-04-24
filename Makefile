# ========================================
# AlphaZero Producer-Consumer Makefile
# ========================================
# Edit these variables before running!

# CONFIGURACIÓN PRINCIPAL
USER_A ?= gmunserver
HOST_A ?= 192.168.50.101
USER_B ?= gmunserver
HOST_B ?= 192.168.50.102

# Parámetros de entrenamiento
BITS ?= 4
HEIGHT ?= 4
N_SIM_PRODUCER ?= 100
N_SIM_CONSUMER ?= 0
GAMES_PER_ITER ?= 20
TRAIN_STEPS ?= 60
BATCH_SIZE ?= 256
ITERATIONS_PRODUCER ?= 50
ITERATIONS_CONSUMER ?= 100

# Directorios
PROJECT_DIR ?= ~/MCTS_CUCA_CACA
GAMES_DIR_A ?= games_prod
GAMES_DIR_B ?= games_all
CHECKPOINT_DIR ?= alphazero/checkpoints

# Timings
SYNC_INTERVAL ?= 300
CONSUMER_SLEEP ?= 1800

# Colors para output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help setup validate sync-code start-producer start-consumer stop-producer stop-consumer \
        status monitor-producer monitor-consumer monitor-all inspect metrics cleanup

help:
	@echo "$(GREEN)=== AlphaZero Producer-Consumer Makefile ===$(NC)"
	@echo ""
	@echo "$(YELLOW)CONFIGURACIÓN (edita estas variables):$(NC)"
	@echo "  USER_A=$(USER_A), HOST_A=$(HOST_A)"
	@echo "  USER_B=$(USER_B), HOST_B=$(HOST_B)"
	@echo "  BITS=$(BITS), N_SIM_PRODUCER=$(N_SIM_PRODUCER), N_SIM_CONSUMER=$(N_SIM_CONSUMER)"
	@echo ""
	@echo "$(YELLOW)TARGETS PRINCIPALES:$(NC)"
	@echo "  make setup              - Preparar ambos workstations"
	@echo "  make validate           - Verificar configuración y conexiones SSH"
	@echo "  make sync-code          - Sincronizar código mejorado (trainer.py)"
	@echo "  make start-producer     - Iniciar WORKSTATION A (generador de juegos)"
	@echo "  make start-consumer     - Iniciar WORKSTATION B (entrenador)"
	@echo "  make stop-producer      - Parar productor"
	@echo "  make stop-consumer      - Parar consumidor"
	@echo ""
	@echo "$(YELLOW)MONITOREO:$(NC)"
	@echo "  make status             - Ver estado de ambos procesos"
	@echo "  make monitor-producer   - Ver logs del productor"
	@echo "  make monitor-consumer   - Ver logs del consumidor"
	@echo "  make monitor-all        - Ver logs de ambos simultáneamente"
	@echo ""
	@echo "$(YELLOW)ANÁLISIS:$(NC)"
	@echo "  make metrics            - Mostrar métricas de mejora"
	@echo "  make inspect            - Generar visualización del buffer"
	@echo ""
	@echo "$(YELLOW)LIMPIEZA:$(NC)"
	@echo "  make cleanup            - Parar procesos y limpiar logs"
	@echo ""

# ========================================
# VALIDACIÓN Y SETUP
# ========================================

validate:
	@echo "$(GREEN)[*] Validando configuración...$(NC)"
	@echo "WORKSTATION A: $(USER_A)@$(HOST_A)"
	@echo "WORKSTATION B: $(USER_B)@$(HOST_B)"
	@echo ""
	@echo "$(YELLOW)[*] Probando SSH a A...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "echo $(GREEN)[+] Conexión OK$(NC)" || \
		(echo "$(RED)[!] No se pudo conectar a $(USER_A)@$(HOST_A)$(NC)" && exit 1)
	@echo "$(YELLOW)[*] Probando SSH a B...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "echo $(GREEN)[+] Conexión OK$(NC)" || \
		(echo "$(RED)[!] No se pudo conectar a $(USER_B)@$(HOST_B)$(NC)" && exit 1)
	@echo ""
	@echo "$(YELLOW)[*] Verificando proyecto en A...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "test -f $(PROJECT_DIR)/alphazero/main.py && \
		echo $(GREEN)[+] Proyecto encontrado$(NC) || \
		(echo $(RED)[!] Proyecto no encontrado$(NC) && exit 1)"
	@echo "$(YELLOW)[*] Verificando proyecto en B...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "test -f $(PROJECT_DIR)/alphazero/main.py && \
		echo $(GREEN)[+] Proyecto encontrado$(NC) || \
		(echo $(RED)[!] Proyecto no encontrado$(NC) && exit 1)"
	@echo ""
	@echo "$(GREEN)[+] ¡Validación completa!$(NC)"

setup: validate
	@echo "$(GREEN)[*] Configurando WORKSTATION A...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "\
		cd $(PROJECT_DIR) && \
		mkdir -p $(GAMES_DIR_A) $(CHECKPOINT_DIR) alphazero/logs && \
		echo $(GREEN)[+] Carpetas creadas en A$(NC)"
	@echo "$(GREEN)[*] Configurando WORKSTATION B...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "\
		cd $(PROJECT_DIR) && \
		mkdir -p $(GAMES_DIR_B) $(CHECKPOINT_DIR) alphazero/logs && \
		echo $(GREEN)[+] Carpetas creadas en B$(NC)"
	@echo "$(GREEN)[+] Setup completado!$(NC)"

sync-code:
	@echo "$(GREEN)[*] Sincronizando código mejorado a B...$(NC)"
	@rsync -avz $(PROJECT_DIR)/alphazero/training/trainer.py \
		$(USER_B)@$(HOST_B):$(PROJECT_DIR)/alphazero/training/
	@echo "$(GREEN)[+] Código sincronizado!$(NC)"

# ========================================
# INICIAR PROCESOS
# ========================================

start-producer: validate
	@echo "$(GREEN)[*] Iniciando PRODUCER en $(USER_A)@$(HOST_A)...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "cat > /tmp/producer.sh << 'PRODEOF'\n\
#!/bin/bash\n\
cd $(PROJECT_DIR)\n\
while true; do\n\
  echo \"[$(date)] Descargando checkpoint del consumer...\"\n\
  LATEST=\$$(ssh $(USER_B)@$(HOST_B) \"ls -t $(PROJECT_DIR)/$(CHECKPOINT_DIR)/checkpoint_*.pt 2>/dev/null | head -1\")\n\
  if [ ! -z \"\$$LATEST\" ]; then\n\
    rsync -avz $(USER_B)@$(HOST_B):\$$LATEST ./checkpoints_producer/ 2>/dev/null\n\
    CHECKPOINT=\"./checkpoints_producer/\$$(basename \$$LATEST)\"\n\
    python3 alphazero/main.py --bits $(BITS) --n-sim $(N_SIM_PRODUCER) --iterations $(ITERATIONS_PRODUCER) --games-per-iter $(GAMES_PER_ITER) --train-steps 0 --resume \$$CHECKPOINT --games-dir ./$(GAMES_DIR_A) --device cuda\n\
  else\n\
    python3 alphazero/main.py --bits $(BITS) --n-sim $(N_SIM_PRODUCER) --iterations $(ITERATIONS_PRODUCER) --games-per-iter $(GAMES_PER_ITER) --train-steps 0 --games-dir ./$(GAMES_DIR_A) --device cuda\n\
  fi\n\
  sleep $(SYNC_INTERVAL)\n\
done\n\
PRODEOF\n\
chmod +x /tmp/producer.sh"
	@ssh $(USER_A)@$(HOST_A) "nohup bash /tmp/producer.sh > /tmp/producer.log 2>&1 &"
	@echo "$(GREEN)[+] Producer iniciado. Ver logs: make monitor-producer$(NC)"

start-consumer: validate sync-code
	@echo "$(GREEN)[*] Iniciando CONSUMER en $(USER_B)@$(HOST_B)...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "cat > /tmp/consumer.sh << 'CONEOF'\n\
#!/bin/bash\n\
cd $(PROJECT_DIR)\n\
while true; do\n\
  echo \"[$$(date)] Sincronizando juegos...\"\n\
  rsync -avz --delete-after $(USER_A)@$(HOST_A):$(PROJECT_DIR)/$(GAMES_DIR_A)/ ./$(GAMES_DIR_B)/ 2>/dev/null\n\
  GAME_COUNT=$$(ls -1 $(GAMES_DIR_B)/*.pt 2>/dev/null | wc -l)\n\
  echo \"[$$(date)] Total juegos: $$GAME_COUNT\"\n\
  if [ $$GAME_COUNT -gt 100 ]; then\n\
    python3 alphazero/main.py --bits $(BITS) --n-sim $(N_SIM_CONSUMER) --iterations $(ITERATIONS_CONSUMER) --train-steps $(TRAIN_STEPS) --load-games ./$(GAMES_DIR_B) --batch-size $(BATCH_SIZE) --device cuda\n\
  fi\n\
  sleep $(CONSUMER_SLEEP)\n\
done\n\
CONEOF\n\
chmod +x /tmp/consumer.sh"
	@ssh $(USER_B)@$(HOST_B) "nohup bash /tmp/consumer.sh > /tmp/consumer.log 2>&1 &"
	@echo "$(GREEN)[+] Consumer iniciado. Ver logs: make monitor-consumer$(NC)"

# ========================================
# PARAR PROCESOS
# ========================================

stop-producer:
	@echo "$(YELLOW)[*] Parando producer...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "pkill -f 'python3 alphazero/main.py'; pkill -f 'producer.sh'" || true
	@echo "$(GREEN)[+] Producer parado$(NC)"

stop-consumer:
	@echo "$(YELLOW)[*] Parando consumer...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "pkill -f 'python3 alphazero/main.py'; pkill -f 'consumer.sh'" || true
	@echo "$(GREEN)[+] Consumer parado$(NC)"

# ========================================
# MONITOREO
# ========================================

status:
	@echo "$(GREEN)=== ESTADO PRODUCER ===$(NC)"
	@ssh $(USER_A)@$(HOST_A) "ps aux | grep -E 'python3|producer.sh' | grep -v grep || echo 'No running'" || true
	@echo ""
	@echo "$(GREEN)=== ESTADO CONSUMER ===$(NC)"
	@ssh $(USER_B)@$(HOST_B) "ps aux | grep -E 'python3|consumer.sh' | grep -v grep || echo 'No running'" || true

monitor-producer:
	@echo "$(GREEN)[*] Monitoreando PRODUCER (Ctrl+C para salir)...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "tail -f /tmp/producer.log"

monitor-consumer:
	@echo "$(GREEN)[*] Monitoreando CONSUMER (Ctrl+C para salir)...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "tail -f /tmp/consumer.log"

monitor-all:
	@echo "$(GREEN)[*] Para ver ambos logs, abre dos terminales:$(NC)"
	@echo "  Terminal 1: make monitor-producer"
	@echo "  Terminal 2: make monitor-consumer"
	@echo ""
	@echo "$(YELLOW)Monitor GPU en PRODUCER:$(NC)"
	@ssh $(USER_A)@$(HOST_A) "watch -n 2 nvidia-smi" || true

# ========================================
# ANÁLISIS Y MÉTRICAS
# ========================================

metrics:
	@echo "$(GREEN)=== MÉTRICAS DE MEJORA ===$(NC)"
	@echo ""
	@echo "$(YELLOW)JUEGOS GENERADOS (Producer):$(NC)"
	@ssh $(USER_A)@$(HOST_A) "ls -1 $(PROJECT_DIR)/$(GAMES_DIR_A)/*.pt 2>/dev/null | wc -l || echo '0'" | xargs echo "  Total:"
	@echo ""
	@echo "$(YELLOW)JUEGOS DESCARGADOS (Consumer):$(NC)"
	@ssh $(USER_B)@$(HOST_B) "ls -1 $(PROJECT_DIR)/$(GAMES_DIR_B)/*.pt 2>/dev/null | wc -l || echo '0'" | xargs echo "  Total:"
	@echo ""
	@echo "$(YELLOW)CHECKPOINTS GENERADOS:$(NC)"
	@ssh $(USER_B)@$(HOST_B) "ls -lth $(PROJECT_DIR)/$(CHECKPOINT_DIR)/checkpoint_*.pt 2>/dev/null | head -5 | awk '{print \"  \" \$$9, \"-\", \$$5, \"bytes\"}' || echo '  Ninguno'" || true
	@echo ""
	@echo "$(YELLOW)TAMAÑO TOTAL (Producer):$(NC)"
	@ssh $(USER_A)@$(HOST_A) "du -sh $(PROJECT_DIR)/$(GAMES_DIR_A) 2>/dev/null | awk '{print \"  \" \$$1}'" || true
	@echo ""
	@echo "$(YELLOW)TAMAÑO TOTAL (Consumer):$(NC)"
	@ssh $(USER_B)@$(HOST_B) "du -sh $(PROJECT_DIR)/$(GAMES_DIR_B) 2>/dev/null | awk '{print \"  \" \$$1}'" || true

inspect:
	@echo "$(GREEN)[*] Generando visualización del buffer...$(NC)"
	@ssh $(USER_B)@$(HOST_B) "cd $(PROJECT_DIR) && \
		python3 alphazero/main.py --bits $(BITS) \
		--load-games ./$(GAMES_DIR_B) \
		--inspect ./buffer_snapshot_$$(date +%s).png \
		--inspect-n 8"
	@echo "$(GREEN)[+] Visualización guardada$(NC)"

# ========================================
# LIMPIEZA
# ========================================

cleanup: stop-producer stop-consumer
	@echo "$(YELLOW)[*] Limpiando procesos y logs...$(NC)"
	@ssh $(USER_A)@$(HOST_A) "rm -f /tmp/producer.log /tmp/producer.sh" || true
	@ssh $(USER_B)@$(HOST_B) "rm -f /tmp/consumer.log /tmp/consumer.sh" || true
	@echo "$(GREEN)[+] Limpieza completada$(NC)"

# ========================================
# QUICK START
# ========================================

quick-start: setup sync-code
	@echo ""
	@echo "$(GREEN)=== QUICK START ===$(NC)"
	@echo "$(YELLOW)Paso 1: Inicia el producer$(NC)"
	@echo "  make start-producer"
	@echo ""
	@echo "$(YELLOW)Paso 2: En otra terminal, inicia el consumer$(NC)"
	@echo "  make start-consumer"
	@echo ""
	@echo "$(YELLOW)Paso 3: Monitorea el progreso$(NC)"
	@echo "  make monitor-all"
	@echo ""
	@echo "$(YELLOW)Paso 4: Visualiza métricas$(NC)"
	@echo "  make metrics"
	@echo ""
