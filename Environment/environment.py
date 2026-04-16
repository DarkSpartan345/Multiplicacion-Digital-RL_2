import gymnasium as gym
import numpy as np
import pygame
import random
import copy
from .env_base import BinaryMathEnv


class BinaryMathEnvSecuencial(BinaryMathEnv):
    """
    Entorno de Gym personalizado para rellenar tabla de productos parciales.
    Versión con renderizado visual de la tabla.
    """

    def __init__(self, render_mode=None, Bits=8, Proof=4, height=8, maxi=100):
        super().__init__(Bits=Bits, Proof=Proof, height=height, maxi=maxi)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        pygame.init()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 16)

        # Dimensiones de la pantalla
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 700

        # Colores
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'GRAY': (200, 200, 200),
            'BLUE': (100, 150, 255),
            'RED': (255, 100, 100),
            'GREEN': (100, 255, 100),
            'YELLOW': (255, 255, 100),
        }

    def reset(self, seed=None, options=None):
        """Reiniciar el entorno a su estado inicial."""
        observation = super().reset(seed=seed)
        self.window = None
        if self.render_mode == 'human':
            self._render_frame()
        return observation

    def render(self):
        """Método de renderizado para diferentes modos."""
        if self.render_mode is None:
            return

        if self.render_mode == 'human':
            self._render_frame()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()

    def _render_frame(self):
        """Renderizar frame mostrando la tabla de productos parciales y el Verilog generado."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption('Rellenar Tabla de Productos Parciales - Con Visualización de Verilog')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill(self.COLORS['WHITE'])

        # Título
        title = self.font_large.render(self.phase_names[0], True, self.COLORS['BLUE'])
        self.window.blit(title, (20, 10))

        # ============ SECCIÓN IZQUIERDA: TABLA DE PRODUCTOS PARCIALES ============
        cell_width = 60
        cell_height = 25
        start_x = 20
        start_y = 50

        # Encabezados de columnas
        for col in range(2 * self.Bits):
            if col % 2 == 0:  # Solo mostrar cada dos columnas para no saturar
                col_text = self.font_small.render(f"{col}", True, self.COLORS['BLACK'])
                x = start_x + col * cell_width + 5
                y = start_y - 20
                self.window.blit(col_text, (x, y))

        # Renderizar celdas de la tabla
        for row in range(self.height):
            row_text = self.font_small.render(f"R{row}", True, self.COLORS['BLACK'])
            self.window.blit(row_text, (start_x - 25, start_y + row * cell_height + 5))

            for col in range(2 * self.Bits):
                index = row * (2 * self.Bits) + col
                x = start_x + col * cell_width
                y = start_y + row * cell_height

                # Determinar color de celda
                if index == self.cursor_position:
                    border_color = self.COLORS['RED']
                    fill_color = self.COLORS['YELLOW']
                elif self.suma_grid[index] != ' ':
                    border_color = self.COLORS['BLUE']
                    fill_color = self.COLORS['GREEN']
                else:
                    border_color = self.COLORS['GRAY']
                    fill_color = self.COLORS['WHITE']

                # Dibujar celda
                pygame.draw.rect(self.window, fill_color, (x, y, cell_width, cell_height))
                pygame.draw.rect(self.window, border_color, (x, y, cell_width, cell_height), 1)

                # Dibujar contenido (versión comprimida)
                cell_str = str(self.suma_grid[index])
                if len(cell_str) > 8:
                    cell_str = cell_str[:7] + "."
                cell_text = self.font_small.render(cell_str, True, self.COLORS['BLACK'])
                text_rect = cell_text.get_rect(center=(x + cell_width // 2, y + cell_height // 2))
                self.window.blit(cell_text, text_rect)

        # ============ INFORMACIÓN DE ESTADO (IZQUIERDA) ============
        info_y = start_y + self.height * cell_height + 20
        cursor_info = self.font_small.render(
            f"Cursor: {self.cursor_position}/{self.CC}",
            True, self.COLORS['BLACK']
        )
        self.window.blit(cursor_info, (start_x, info_y))

        casillas_rellenas = sum(1 for cell in self.suma_grid if cell != ' ')
        filled_info = self.font_small.render(
            f"Rellenas: {casillas_rellenas}/{self.CC}",
            True, self.COLORS['BLACK']
        )
        self.window.blit(filled_info, (start_x, info_y + 20))

        reward_info = self.font_small.render(
            f"Reward: {self.reward:.2f}",
            True, self.COLORS['BLACK']
        )
        self.window.blit(reward_info, (start_x, info_y + 40))

        # ============ SECCIÓN DERECHA: INFORMACIÓN DE VERILOG ============
        verilog_x = 500
        verilog_y = 50

        # Título de sección
        verilog_title = self.font_medium.render("Información del Circuito", True, self.COLORS['BLUE'])
        self.window.blit(verilog_title, (verilog_x, verilog_y))

        verilog_info_y = verilog_y + 30

        # Productos parciales únicos
        unique_products = list(set([s for s in self.suma_grid if s not in [' ', '0', '1']]))
        products_info = self.font_small.render(
            f"Productos Únicos: {len(unique_products)}",
            True, self.COLORS['BLACK']
        )
        self.window.blit(products_info, (verilog_x, verilog_info_y))

        # Listar productos únicos (solo los primeros 5 para no saturar)
        for idx, product in enumerate(unique_products[:5]):
            product_short = product[:30] + "..." if len(product) > 30 else product
            product_text = self.font_small.render(f"  - {product_short}", True, self.COLORS['BLACK'])
            self.window.blit(product_text, (verilog_x, verilog_info_y + 20 + idx * 18))

        if len(unique_products) > 5:
            more_text = self.font_small.render(f"  ... y {len(unique_products) - 5} más", True, self.COLORS['BLACK'])
            self.window.blit(more_text, (verilog_x, verilog_info_y + 20 + 5 * 18))

        # Mostrar métricas del circuito si están disponibles
        metrics_y = verilog_info_y + 150
        if hasattr(self, 'last_metrics'):
            metrics = self.last_metrics
            if 'error_mean' in metrics:
                error_info = self.font_small.render(
                    f"Error Funcional: {metrics['error_mean']:.4f}",
                    True, self.COLORS['BLACK']
                )
                self.window.blit(error_info, (verilog_x, metrics_y))

                gates_info = self.font_small.render(
                    f"Puertas Lógicas: {metrics['circuit_metrics']['logic_gates']}",
                    True, self.COLORS['BLACK']
                )
                self.window.blit(gates_info, (verilog_x, metrics_y + 20))

                wires_info = self.font_small.render(
                    f"Conexiones: {metrics['circuit_metrics']['wires']}",
                    True, self.COLORS['BLACK']
                )
                self.window.blit(wires_info, (verilog_x, metrics_y + 40))

                operands_info = self.font_small.render(
                    f"Operandos: {metrics['circuit_metrics']['operand_count']}",
                    True, self.COLORS['BLACK']
                )
                self.window.blit(operands_info, (verilog_x, metrics_y + 60))

                reward_final = self.font_small.render(
                    f"Reward Final: {metrics['final_reward']:.2f}",
                    True, self.COLORS['BLACK']
                )
                self.window.blit(reward_final, (verilog_x, metrics_y + 80))

        # ============ SECCIÓN CÓDIGO VERILOG (ABAJO DERECHA) ============
        verilog_code_y = verilog_y + 350
        verilog_code_title = self.font_medium.render("Vista Previa del Verilog Generado", True, self.COLORS['BLUE'])
        self.window.blit(verilog_code_title, (verilog_x, verilog_code_y))

        # Mostrar código Verilog si está disponible
        if hasattr(self, 'last_verilog_code'):
            code_lines = self.last_verilog_code.split('\n')[:8]  # Primeras 8 líneas
            for idx, line in enumerate(code_lines):
                if len(line) > 50:
                    line = line[:47] + "..."
                code_text = self.font_small.render(line, True, self.COLORS['BLACK'])
                self.window.blit(code_text, (verilog_x, verilog_code_y + 25 + idx * 15))

        # Actualizar pantalla
        pygame.display.flip()
        self.clock.tick(30)

    def _render_rgb_array(self):
        """Renderizar a array RGB sin mostrar ventana"""
        self._render_frame()
        if self.window is not None:
            # Convertir superficie de pygame a array NumPy
            return pygame.surfarray.array3d(pygame.display.get_surface()).transpose((2, 0, 1))
        return None

    def clone(self):
        """
        Crea una copia profunda del entorno secuencial (sin recursos pygame).
        Los recursos pygame se reinicializan en el clon.

        Returns:
            BinaryMathEnvSecuencial: Nuevo entorno clonado con el estado idéntico.
        """
        # Guardar referencias de recursos pygame
        window_backup = self.window
        clock_backup = self.clock
        font_large_backup = self.font_large
        font_medium_backup = self.font_medium
        font_small_backup = self.font_small

        # Limpiar recursos pygame temporalmente
        self.window = None
        self.clock = None
        self.font_large = None
        self.font_medium = None
        self.font_small = None

        try:
            # Hacer deepcopy sin los recursos pygame
            cloned_env = copy.deepcopy(self)
        finally:
            # Restaurar recursos en el original
            self.window = window_backup
            self.clock = clock_backup
            self.font_large = font_large_backup
            self.font_medium = font_medium_backup
            self.font_small = font_small_backup

        # Reinicializar recursos pygame en el clon
        pygame.init()
        cloned_env.font_large = pygame.font.Font(None, 36)
        cloned_env.font_medium = pygame.font.Font(None, 24)
        cloned_env.font_small = pygame.font.Font(None, 16)

        return cloned_env

    def get_state(self):
        """
        Obtiene el estado actual del entorno secuencial como un diccionario.
        Excluye recursos pygame que no pueden serializarse.

        Returns:
            dict: Diccionario con todas las variables de estado importantes.
        """
        state = super().get_state()
        # Agregar configuración de render mode específica
        state['render_mode'] = self.render_mode
        return state

    def set_state(self, state):
        """
        Restaura el estado del entorno secuencial desde un diccionario.
        Preserva la configuración de renderizado.

        Args:
            state (dict): Diccionario con el estado obtenido de get_state().
        """
        super().set_state(state)
        if 'render_mode' in state:
            self.render_mode = state['render_mode']
        # Reinicializar recursos pygame
        self.window = None
        self.clock = None
        pygame.init()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 16)