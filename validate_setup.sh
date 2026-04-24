#!/bin/bash
# ========================================
# Validador rápido de configuración
# ========================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AlphaZero Setup Validator${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Colores y variables
PROJECT_DIR="MCTS_CUCA_CACA"
REQUIRED_FILES=(
    "alphazero/main.py"
    "alphazero/training/trainer.py"
    "alphazero/training/self_play.py"
    "alphazero/mcts/search.py"
    "alphazero/model/network.py"
    "Environment/env_cuda.py"
)

# 1. Verificar que está en el directorio correcto
echo -e "${YELLOW}[1] Verificando directorio actual...${NC}"
if [[ ! $PWD == *"$PROJECT_DIR"* ]]; then
    echo -e "${RED}[!] Error: debe estar en $PROJECT_DIR${NC}"
    echo -e "${RED}    Directorio actual: $PWD${NC}"
    exit 1
fi
echo -e "${GREEN}[+] OK: en directorio correcto${NC}"
echo ""

# 2. Verificar archivos necesarios
echo -e "${YELLOW}[2] Verificando archivos necesarios...${NC}"
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}[+] $file${NC}"
    else
        echo -e "${RED}[!] $file NO ENCONTRADO${NC}"
        exit 1
    fi
done
echo -e "${GREEN}[+] Todos los archivos OK${NC}"
echo ""

# 3. Verificar Python y dependencias
echo -e "${YELLOW}[3] Verificando Python y librerías...${NC}"
python3 -c "import torch; print('[+] PyTorch', torch.__version__)" || \
    { echo -e "${RED}[!] PyTorch no instalado${NC}"; exit 1; }
python3 -c "import numpy; print('[+] NumPy', numpy.__version__)" || \
    { echo -e "${RED}[!] NumPy no instalado${NC}"; exit 1; }
python3 -c "from tensorboard import program; print('[+] TensorBoard OK')" || \
    { echo -e "${RED}[!] TensorBoard no instalado${NC}"; exit 1; }
echo ""

# 4. Test rápido: importar módulos
echo -e "${YELLOW}[4] Verificando que módulos se importan...${NC}"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from alphazero.model.encoder import StateEncoder
    print("\033[0;32m[+] StateEncoder\033[0m")
    from alphazero.model.network import AlphaZeroNet
    print("\033[0;32m[+] AlphaZeroNet\033[0m")
    from alphazero.mcts.search import MCTSSearch
    print("\033[0;32m[+] MCTSSearch\033[0m")
    from alphazero.training.trainer import AlphaZeroTrainer
    print("\033[0;32m[+] AlphaZeroTrainer\033[0m")
    print("\033[0;32m[+] Todos los módulos se importan correctamente\033[0m")
except Exception as e:
    print(f"\033[0;31m[!] Error: {e}\033[0m")
    sys.exit(1)
PYEOF
echo ""

# 5. Verificar GPU
echo -e "${YELLOW}[5] Verificando GPU...${NC}"
python3 << 'GPUEOF'
import torch
if torch.cuda.is_available():
    print(f"\033[0;32m[+] GPU disponible: {torch.cuda.get_device_name(0)}\033[0m")
    print(f"\033[0;32m    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\033[0m")
else:
    print("\033[1;33m[!] CUDA no disponible (usará CPU)\033[0m")
GPUEOF
echo ""

# 6. Test rápido: crear red pequeña
echo -e "${YELLOW}[6] Test rápido: crear y ejecutar red...${NC}"
python3 << 'NETTEST'
import sys
sys.path.insert(0, '.')
import torch
from Environment.env_cuda import BinaryMathEnvCUDA
from alphazero.model.encoder import StateEncoder
from alphazero.model.network import AlphaZeroNet

try:
    env = BinaryMathEnvCUDA(Bits=4, height=4, n_envs=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    encoder = StateEncoder(env)
    net = AlphaZeroNet.from_env(env, d_model=32, n_filters=64, n_res=3)

    # Forward pass - grid shape debe ser (batch, height, width)
    grid = torch.randint(0, 5, (1, 4, 8), dtype=torch.int64)
    policy, value = net(grid)

    print(f"\033[0;32m[+] Red ejecutada exitosamente\033[0m")
    print(f"\033[0;32m    Policy shape: {policy.shape}\033[0m")
    print(f"\033[0;32m    Value shape: {value.shape}\033[0m")
except Exception as e:
    print(f"\033[0;31m[!] Error en red: {e}\033[0m")
    sys.exit(1)
NETTEST
echo ""

# 7. Verificar trainer.py mejorado
echo -e "${YELLOW}[7] Verificando trainer.py...${NC}"
if grep -q "if self.n_simulations > 0:" alphazero/training/trainer.py; then
    echo -e "${GREEN}[+] trainer.py tiene corrección de n_sim=0${NC}"
else
    echo -e "${RED}[!] trainer.py no tiene corrección - ejecuta: rsync del código${NC}"
fi
echo ""

# Resumen
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ VALIDACIÓN COMPLETA${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Próximos pasos:${NC}"
echo "1. Edita Makefile con tus credenciales SSH"
echo "2. Ejecuta: make validate"
echo "3. Ejecuta: make setup"
echo "4. En Terminal 1: make start-producer"
echo "5. En Terminal 2: make start-consumer"
echo "6. Monitorea: make metrics"
echo ""
