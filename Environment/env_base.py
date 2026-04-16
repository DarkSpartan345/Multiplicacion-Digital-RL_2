import gymnasium as gym
import numpy as np
import random
from datetime import datetime
import copy


class BinaryMathEnv(gym.Env):
    """
    Entorno de Gym personalizado para rellenar tabla de productos parciales.
    El agente debe seleccionar el valor correcto (producto parcial, 0, 1) para cada casilla.
    """

    def __init__(self, Bits=8, Proof=4, height=8, maxi=100):
        super().__init__()  # Inicializar clase base
        # Configuración de entorno
        self.height = height
        self.Proof = Proof
        self.Bits = Bits
        self.maxi = maxi
        self.it = 0  # Iteración actual (número de operaciones)
        self.CP = 0  # Contador de pasos

        # Grid de productos parciales
        self.CC = height * 2 * self.Bits  # Total de casillas
        self.suma_grid = [' '] * self.CC  # Estado actual de la tabla
        self.grid_size = 2 * self.Bits
        self.reward = 0

        # Productos parciales posibles: (A[i] & B[j]) para todos i, j
        # Más negaciones (~) y constantes (0, 1)
        self._generate_possible_actions()

        # Espacios de Gym
        # Action space: seleccionar uno de los productos parciales posibles
        self.action_space = gym.spaces.Discrete(len(self.possible_actions))
        # Observation space: (cursor_position, numero_de_casillas_rellenas)
        self.observation_space = gym.spaces.MultiDiscrete([self.CC, self.CC + 1])

        # Estados del juego
        self.current_phase = 2  # Solo fase de productos parciales
        self.cursor_position = 0  # Posición actual en la tabla
        self.phase_names = ("Rellenar Tabla de Productos Parciales",)
        self.min_error = 0.9

    def _generate_possible_actions(self):
        """Genera la lista de acciones posibles: productos parciales, negaciones, 0, 1"""
        self.possible_actions = ['0', '1']

        # Agregar todos los productos parciales (A[i] & B[j]) para todos i, j
        for i in range(self.Bits):
            for j in range(self.Bits):
                self.possible_actions.append(f'(A[{i}]&B[{j}])')
                self.possible_actions.append(f'(~A[{i}]&B[{j}])')
                self.possible_actions.append(f'(A[{i}]&~B[{j}])')
                self.possible_actions.append(f'(~A[{i}]&~B[{j}])')
    def step(self, action, arch_multiplier=None, arch_multipliermax=None,
             arch_multiplier_8bit_tb=None, arch_simv=None):
        """
        Ejecutar un paso en el entorno.
        El agente selecciona un valor (producto parcial, 0 o 1) para la casilla actual.
        Recompensa: 0 durante pasos intermedios.
        Recompensa final: basada en error del circuito cuando se completa la tabla.
        """
        # Usar rutas por defecto relativas al proyecto
        from pathlib import Path
        if arch_multiplier is None:
            project_root = Path(__file__).parent.parent
            arch_multiplier = str(project_root / 'verilog' / 'multiplier.v')
        if arch_multipliermax is None:
            project_root = Path(__file__).parent.parent
            arch_multipliermax = str(project_root / 'verilog' / 'multipliermax.v')
        if arch_multiplier_8bit_tb is None:
            project_root = Path(__file__).parent.parent
            arch_multiplier_8bit_tb = str(project_root / 'verilog' / 'multiplier_8bit_tb.v')
        if arch_simv is None:
            project_root = Path(__file__).parent.parent
            arch_simv = str(project_root / 'verilog' / 'simv')

        terminated = False
        truncated = False
        self.reward = 0.0  # Sin recompensa intermedia (pasos fijos por tamaño de tabla)
        self.CP += 1

        # Validar que el action sea válido
        if action < 0 or action >= len(self.possible_actions):
            truncated = True
        # Verificar si el episodio ya terminó
        elif self.cursor_position >= self.CC:
            # Episodio ya completado, no hacer nada
            truncated = True
        else:
            selected_action = self.possible_actions[action]

            # Rellenar la casilla actual en orden
            self.suma_grid[self.cursor_position] = selected_action
            self.cursor_position += 1

            # Verificar si completó la tabla
            if self.cursor_position >= self.CC:
                self.it += 1
                # Calcular recompensa final basada en error del circuito
                terminated = self.closed(arch_multiplier=arch_multiplier,
                                        arch_multiplier_8bit_tb=arch_multiplier_8bit_tb,
                                        arch_simv=arch_simv, arch_multipliermax=arch_multipliermax)
                # self.reward será establecida por closed()


        return self._get_observation(), self.reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reiniciar el entorno a su estado inicial."""
        super().reset(seed=seed)
        self.cursor_position = 0
        self.suma_grid = [' '] * self.CC
        self.it = 0
        self.CP = 0
        self.reward = 0

        return self._get_observation(), {}

    def clone(self):
        """
        Crea una copia profunda del entorno con el mismo estado actual.
        Útil para MCTS y exploración de diferentes caminos.

        Returns:
            BinaryMathEnv: Nuevo entorno clonado con el estado idéntico.
        """
        cloned_env = copy.deepcopy(self)
        return cloned_env

    def get_state(self):
        """
        Obtiene el estado actual del entorno como un diccionario.
        Permite serializar y guardar el estado para restaurarlo después.

        Returns:
            dict: Diccionario con todas las variables de estado importantes.
        """
        state = {
            'cursor_position': self.cursor_position,
            'suma_grid': copy.deepcopy(self.suma_grid),
            'it': self.it,
            'CP': self.CP,
            'reward': self.reward,
            'min_error': self.min_error,
            'current_phase': self.current_phase,
            'height': self.height,
            'Proof': self.Proof,
            'Bits': self.Bits,
            'maxi': self.maxi,
            'CC': self.CC,
            'grid_size': self.grid_size,
        }
        if hasattr(self, 'last_metrics'):
            state['last_metrics'] = copy.deepcopy(self.last_metrics)
        if hasattr(self, 'last_verilog_code'):
            state['last_verilog_code'] = self.last_verilog_code
        return state

    def set_state(self, state):
        """
        Restaura el estado del entorno desde un diccionario previamente guardado.
        Permite volver a un estado anterior (útil para MCTS).

        Args:
            state (dict): Diccionario con el estado obtenido de get_state().
        """
        self.cursor_position = state['cursor_position']
        self.suma_grid = copy.deepcopy(state['suma_grid'])
        self.it = state['it']
        self.CP = state['CP']
        self.reward = state['reward']
        self.min_error = state['min_error']
        self.current_phase = state['current_phase']

        if 'last_metrics' in state:
            self.last_metrics = copy.deepcopy(state['last_metrics'])
        if 'last_verilog_code' in state:
            self.last_verilog_code = state['last_verilog_code']

    def _get_observation(self):
        """
        Obtener el estado actual del entorno.
        Retorna: array numpy de (posición_cursor, número_casillas_rellenas)
        """
        casillas_rellenas = sum(1 for cell in self.suma_grid if cell != ' ')
        return np.array([self.cursor_position, casillas_rellenas], dtype=np.int32)
    
    def closed(self, arch_multiplier=None,
               arch_multipliermax=None,
               arch_multiplier_8bit_tb=None,
               arch_simv=None):
        """
        Evaluar la solución generada y calcular la recompensa final.
        Recompensa basada en: error funcional + métricas del circuito (opcional).
        """
        # Usar rutas por defecto relativas al proyecto
        from pathlib import Path
        if arch_multiplier is None:
            project_root = Path(__file__).parent.parent
            arch_multiplier = str(project_root / 'verilog' / 'multiplier.v')
        if arch_multipliermax is None:
            project_root = Path(__file__).parent.parent
            arch_multipliermax = str(project_root / 'verilog' / 'multipliermax.v')
        if arch_multiplier_8bit_tb is None:
            project_root = Path(__file__).parent.parent
            arch_multiplier_8bit_tb = str(project_root / 'verilog' / 'multiplier_8bit_tb.v')
        if arch_simv is None:
            project_root = Path(__file__).parent.parent
            arch_simv = str(project_root / 'verilog' / 'simv')

        terminated = True
        try:
            test_cases, results = self.generate_verilog(arch_multiplier=arch_multiplier,
                                                        arch_multiplier_8bit_tb=arch_multiplier_8bit_tb,
                                                        arch_simv=arch_simv)

            # Validar que hay resultados
            if len(results) == 0 or len(test_cases) == 0:
                self.reward = -50
                return terminated

            test_cases_results = test_cases[:, 0] * test_cases[:, 1]

            # Calcular error funcional (MSE normalizado)
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs((results - test_cases_results) / (test_cases_results+1e-9))

            error_mean = np.mean(error)
            error_penalty = error_mean * 100  # Penalización por error

            # Calcular métricas del circuito
            circuit_metrics = self._calculate_circuit_metrics(arch_multiplier)

            # Recompensa final: minimizar error + minimizar complejidad
            # Componentes:
            # - Error funcional (primario)
            # - Complejidad del circuito (secundario)
            self.reward = 100 - error_penalty

            # Log de métricas para debugging
            self.last_metrics = {
                'error_mean': error_mean,
                'error_penalty': error_penalty,
                'circuit_metrics': circuit_metrics,
                'final_reward': self.reward
            }

            # Guardar mejor solución
            if error_mean < self.min_error:
                with open(arch_multiplier, 'r') as f:
                    content = f.read()
                with open(arch_multipliermax, 'w') as f:
                    f.write(content)
                self.min_error = error_mean

        except Exception as e:
            print(f"Error en closed: {e}")
            self.reward = -50
            self.last_metrics = {'error': str(e)}

        return terminated

    def _calculate_circuit_metrics(self, arch_multiplier):
        """
        Calcula métricas del circuito como complejidad, número de operaciones, etc.
        """
        metrics = {
            'logic_gates': 0,
            'wires': 0,
            'complexity_penalty': 0,
            'operand_count': 0
        }

        try:
            with open(arch_multiplier, 'r') as f:
                code = f.read()

            # Contar puertas lógicas simuladas (&, |, ~)
            metrics['logic_gates'] = code.count('&') + code.count('|')
            metrics['wires'] = code.count('wire')
            metrics['operand_count'] = len(set([s for s in self.suma_grid if s != ' ']))

            # Penalidad por complejidad: circuitos con muchas operaciones son menos eficientes
            # Normalizamos por tamaño esperado
            expected_gates = self.Bits * self.Bits * 2  # Aproximación
            if metrics['logic_gates'] > expected_gates * 2:
                metrics['complexity_penalty'] = 10  # Penalización moderada
            else:
                metrics['complexity_penalty'] = 0

        except Exception as e:
            print(f"Error calculando métricas: {e}")

        return metrics
    def generate_verilog(self, seed=None, arch_multiplier=None,
                        arch_multiplier_8bit_tb=None, arch_simv=None,
                        test_cases=None):
        """
        Genera código Verilog a partir de la tabla de productos parciales rellenada.
        Elimina productos parciales redundantes (duplicados).
        Almacena el código en self.last_verilog_code para visualización.

        Args:
            test_cases: array (Proof, 2) con casos de prueba. Si None, genera aleatorios.
        """
        import subprocess

        # Usar rutas por defecto relativas al proyecto
        from pathlib import Path
        if arch_multiplier is None:
            project_root = Path(__file__).parent.parent
            arch_multiplier = str(project_root / 'verilog' / 'multiplier.v')
        if arch_multiplier_8bit_tb is None:
            project_root = Path(__file__).parent.parent
            arch_multiplier_8bit_tb = str(project_root / 'verilog' / 'multiplier_8bit_tb.v')
        if arch_simv is None:
            project_root = Path(__file__).parent.parent
            arch_simv = str(project_root / 'verilog' / 'simv')

        # Remodelar grid y obtener productos únicos SIN DUPLICADOS
        suma_grid = np.array(self.suma_grid).reshape(self.height, 2 * self.Bits)

        # Obtener solo productos únicos (eliminar duplicados)
        unique_products = []
        seen = set()
        for s in self.suma_grid:
            if s.strip() != '' and s != ' ' and s not in seen:
                unique_products.append(s)
                seen.add(s)

        multi = unique_products

        # Mapeo de productos a variables wire (sin redundancias)
        suma = {}
        partial_products = []
        for j, product in enumerate(multi):
            wire_name = f"pp{j}"
            partial_products.append(f"    wire pp{j} = {product};")
            suma[product] = wire_name

        # Generar código Verilog
        code_mult = f"""
`timescale 1ns/1ps
module multiplier (
    input [{self.Bits - 1}:0] A,
    input [{self.Bits - 1}:0] B,
    output [{2 * self.Bits - 1}:0] P);

    // Generación de productos parciales (sin redundancias)
"""
        code_mult += "\n".join(partial_products) + "\n\n    // Suma de productos parciales\n"

        columnas = []
        js = []

        for col in range(2 * self.Bits):
            columna_actual = suma_grid[:, col]
            # Obtener valores únicos de esta columna, manteniendo el orden
            valores = []
            seen_col = set()
            for s in columna_actual:
                if s in suma and s not in seen_col:
                    valores.append(suma[s])
                    seen_col.add(s)

            if valores:  # Si hay productos en esta columna
                idx = 2 * self.Bits - col
                js.append(idx)
                suma_expr = " + ".join(valores)
                columnas.append(f"    wire [{self.Bits - 1}:0] columna{idx} = {suma_expr};")

        code_mult += "\n".join(columnas) + "\n"

        if js:
            code_mult += "    assign P = " + " + ".join([f"(columna{i} << {i - 1})" for i in js]) + ";\n"
        else:
            code_mult += "    assign P = 0;\n"

        code_mult += "\nendmodule"

        # Almacenar código para visualización en pygame
        self.last_verilog_code = code_mult

        # Asegurar que el directorio existe
        import os
        mult_dir = os.path.dirname(arch_multiplier)
        if mult_dir and not os.path.exists(mult_dir):
            os.makedirs(mult_dir, exist_ok=True)

        with open(arch_multiplier, 'w') as f:
            f.write(code_mult)

        if seed is not None:
            random.seed(seed)

        # Generar casos de prueba (o usar los proporcionados)
        if test_cases is None:
            test_cases = np.random.randint(1, 2 ** self.Bits, size=(self.Proof, 2))
        else:
            # Convertir a numpy array si es necesario
            test_cases = np.asarray(test_cases)

        # Leer y modificar testbench
        try:
            # Usar ruta relativa al directorio del proyecto
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            template_path = project_root / 'verilog' / 'testbench_template.v'

            with open(template_path, 'r') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: No se encontró testbench_template.v en {template_path}")
            return np.array([]), np.array([])

        content = content.replace("{regsI}", str(self.Bits - 1))
        content = content.replace("{regsO}", str(2 * self.Bits - 1))
        test_text = "\n".join([
            f"""// Caso {i}: Prueba aleatoria {i}
                A = 8'd{test_cases[i, 0]};
                B = 8'd{test_cases[i, 1]};
                #10;
                $display("%d", P);"""
            for i in range(self.Proof)])
        content = content.replace("{Test}", test_text)

        with open(arch_multiplier_8bit_tb, 'w') as f:
            f.write(content)

        # Compilar y simular
        try:
            import os
            from pathlib import Path

            # Asegurar que los archivos existen
            if not os.path.exists(arch_multiplier):
                print(f"⚠️  No se encontró {arch_multiplier} - usando resultado vacío")
                return np.array([]), np.array([])

            if not os.path.exists(arch_multiplier_8bit_tb):
                print(f"⚠️  No se encontró {arch_multiplier_8bit_tb} - usando resultado vacío")
                return np.array([]), np.array([])

            # Usar rutas absolutas
            arch_multiplier = os.path.abspath(arch_multiplier)
            arch_multiplier_8bit_tb = os.path.abspath(arch_multiplier_8bit_tb)
            arch_simv = os.path.abspath(arch_simv)

            # Obtener directorio de trabajo
            work_dir = os.path.dirname(arch_multiplier)

            # Compilar con rutas absolutas
            compile_cmd = ["iverilog", "-o", arch_simv, arch_multiplier, arch_multiplier_8bit_tb]
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=work_dir, timeout=10)

            if compile_result.returncode != 0:
                # Error en compilación - usar valor por defecto sin mostrar error
                # (Los errores de Verilog son esperados en casos de prueba aleatorios)
                return np.array([]), np.array([])

            # Simular
            if not os.path.exists(arch_simv):
                return np.array([]), np.array([])

            simulate = subprocess.run(["vvp", arch_simv], capture_output=True, text=True, cwd=work_dir, timeout=10)
            text = simulate.stdout if simulate.stdout else ""
            results = np.array([int(line.strip()) for line in text.split('\n') if line.strip() and line.strip().isdigit()])
        except subprocess.TimeoutExpired:
            # Simulación tardó demasiado
            results = np.array([])
        except Exception as e:
            # Silenciar errores de simulación (son comunes en pruebas aleatorias)
            results = np.array([])

        return test_cases, results
