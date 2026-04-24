#!/bin/bash
# ========================================
# Generador de Reporte de Métricas
# ========================================

set -e

# Variables de configuración
USER_A="${USER_A:-user_a}"
HOST_A="${HOST_A:-host_a}"
USER_B="${USER_B:-user_b}"
HOST_B="${HOST_B:-host_b}"
PROJECT_DIR="~/MCTS_CUCA_CACA"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

REPORT_FILE="metrics_report_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "=========================================="
    echo "AlphaZero Producer-Consumer Metrics Report"
    echo "=========================================="
    echo "Generated: $(date)"
    echo ""

    # PRODUCER
    echo "${GREEN}=== PRODUCER (WORKSTATION A) ===${NC}"
    echo "Host: $USER_A@$HOST_A"

    echo ""
    echo "Games Generated:"
    ssh $USER_A@$HOST_A "ls -1 $PROJECT_DIR/games_prod/*.pt 2>/dev/null | wc -l || echo '0'" | xargs echo "  Total:"

    echo ""
    echo "Games Size:"
    ssh $USER_A@$HOST_A "du -sh $PROJECT_DIR/games_prod 2>/dev/null | awk '{print \"  \" \$1}'" || echo "  0B"

    echo ""
    echo "Producer Process Status:"
    ssh $USER_A@$HOST_A "ps aux | grep -E 'producer.sh|python3 alphazero' | grep -v grep | wc -l || echo '0'" | xargs echo "  Running processes:"

    echo ""
    echo "Recent Games:"
    ssh $USER_A@$HOST_A "ls -lth $PROJECT_DIR/games_prod/*.pt 2>/dev/null | head -3 | awk '{print \"  \" \$9, \"(\" \$5, \"bytes)\"}'" || echo "  No games yet"

    # CONSUMER
    echo ""
    echo "${GREEN}=== CONSUMER (WORKSTATION B) ===${NC}"
    echo "Host: $USER_B@$HOST_B"

    echo ""
    echo "Games Downloaded:"
    ssh $USER_B@$HOST_B "ls -1 $PROJECT_DIR/games_all/*.pt 2>/dev/null | wc -l || echo '0'" | xargs echo "  Total:"

    echo ""
    echo "Games Size:"
    ssh $USER_B@$HOST_B "du -sh $PROJECT_DIR/games_all 2>/dev/null | awk '{print \"  \" \$1}'" || echo "  0B"

    echo ""
    echo "Checkpoints Generated:"
    ssh $USER_B@$HOST_B "ls -1 $PROJECT_DIR/alphazero/checkpoints/checkpoint_*.pt 2>/dev/null | wc -l || echo '0'" | xargs echo "  Total:"

    echo ""
    echo "Latest Checkpoints:"
    ssh $USER_B@$HOST_B "ls -lth $PROJECT_DIR/alphazero/checkpoints/checkpoint_*.pt 2>/dev/null | head -5 | awk '{print \"  \" \$9, \"(\" \$5, \"bytes, \" \$6, \$7, \$8 \")\"}'" || echo "  No checkpoints yet"

    echo ""
    echo "Consumer Process Status:"
    ssh $USER_B@$HOST_B "ps aux | grep -E 'consumer.sh|python3 alphazero' | grep -v grep | wc -l || echo '0'" | xargs echo "  Running processes:"

    # COMPARATIVES
    echo ""
    echo "${GREEN}=== COMPARATIVAS ===${NC}"

    GAMES_A=$(ssh $USER_A@$HOST_A "ls -1 $PROJECT_DIR/games_prod/*.pt 2>/dev/null | wc -l || echo '0'")
    GAMES_B=$(ssh $USER_B@$HOST_B "ls -1 $PROJECT_DIR/games_all/*.pt 2>/dev/null | wc -l || echo '0'")
    CHECKPOINTS=$(ssh $USER_B@$HOST_B "ls -1 $PROJECT_DIR/alphazero/checkpoints/checkpoint_*.pt 2>/dev/null | wc -l || echo '0'")

    echo "Producer/Consumer Ratio:"
    if [ "$GAMES_B" -gt 0 ]; then
        RATIO=$(echo "scale=2; $GAMES_A / $GAMES_B" | bc)
        echo "  $GAMES_A / $GAMES_B = $RATIO"
    else
        echo "  N/A (consumer no tiene juegos aún)"
    fi

    echo ""
    echo "Estimated Training Progress:"
    if [ "$CHECKPOINTS" -gt 0 ]; then
        CHECKPOINT_ITERS=$((CHECKPOINTS * 10))  # Asumiendo un checkpoint cada 10 iteraciones
        echo "  Checkpoints: $CHECKPOINTS"
        echo "  Estimated Iterations: ~$CHECKPOINT_ITERS"
    else
        echo "  No checkpoints generated yet"
    fi

    # DISK USAGE
    echo ""
    echo "${GREEN}=== CONSUMO DE DISCO ===${NC}"

    echo "Producer Total:"
    ssh $USER_A@$HOST_A "du -sh $PROJECT_DIR/games_prod $PROJECT_DIR/alphazero/checkpoints 2>/dev/null | awk '{s+=\$1} END {print \"  \" s}'" || echo "  0B"

    echo "Consumer Total:"
    ssh $USER_B@$HOST_B "du -sh $PROJECT_DIR/games_all $PROJECT_DIR/alphazero/checkpoints 2>/dev/null | awk '{s+=\$1} END {print \"  \" s}'" || echo "  0B"

    # GPU STATUS
    echo ""
    echo "${GREEN}=== GPU STATUS ===${NC}"

    echo "Producer GPU:"
    ssh $USER_A@$HOST_A "nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo 'N/A'" | xargs echo "  "

    echo ""
    echo "Consumer GPU:"
    ssh $USER_B@$HOST_B "nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo 'N/A'" | xargs echo "  "

    # LOGS
    echo ""
    echo "${GREEN}=== ÚLTIMAS LÍNEAS DE LOG ===${NC}"

    echo ""
    echo "Producer (últimas 5 líneas):"
    ssh $USER_A@$HOST_A "tail -5 /tmp/producer.log 2>/dev/null || echo 'No log found'" | sed 's/^/  /'

    echo ""
    echo "Consumer (últimas 5 líneas):"
    ssh $USER_B@$HOST_B "tail -5 /tmp/consumer.log 2>/dev/null || echo 'No log found'" | sed 's/^/  /'

    echo ""
    echo "=========================================="
    echo "Report saved to: $REPORT_FILE"
    echo "=========================================="

} | tee "$REPORT_FILE"

echo ""
echo "✓ Reporte generado: $REPORT_FILE"
