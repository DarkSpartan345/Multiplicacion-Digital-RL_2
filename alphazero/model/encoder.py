"""State encoder: grid int16 → int64 indices for Embedding layer."""
import torch
from torch import Tensor


class StateEncoder:
    """
    Converts BinaryMathEnvCUDA state tensors into network-ready integer indices.

    Encoding rule:
        -1 (empty cell) → n_actions (the designated EMPTY token index)
        0..n_actions-1 → unchanged

    This keeps the Embedding table at size (n_actions+1, d_model).
    The last row (index n_actions) is the learned "empty cell" embedding.
    """

    def __init__(self, env):
        """Initialize encoder from environment dimensions.

        Args:
            env: BinaryMathEnvCUDA instance or any object with:
                 - height: int (grid rows)
                 - grid_size: int (grid columns = 2*Bits)
                 - n_actions: int (number of valid actions)
                 - CC: int (total cells = height * grid_size)
        """
        self.height = env.height
        self.grid_size = env.grid_size
        self.n_actions = env.n_actions
        self.CC = env.CC

    def encode_batch(self, grids_int16: Tensor) -> Tensor:
        """Encode a batch of grids into integer indices for Embedding lookup.

        Args:
            grids_int16: (batch, CC) int16 tensor from env.suma_grid

        Returns:
            (batch, height, grid_size) int64 tensor with values in [0, n_actions]
        """
        g = grids_int16.long()
        g = g.masked_fill(g == -1, self.n_actions)
        return g.view(-1, self.height, self.grid_size)

    def encode_single(self, env, env_idx: int) -> Tensor:
        """Encode a single environment's grid.

        Args:
            env: BinaryMathEnvCUDA instance
            env_idx: int, which environment slot to extract

        Returns:
            (height, grid_size) int64 tensor
        """
        grid_flat = env.suma_grid[env_idx].unsqueeze(0)
        return self.encode_batch(grid_flat).squeeze(0)

    def decode_grid(self, grid_idx: Tensor, action_names: list[str]) -> list[list[str]]:
        """Convert encoded grid back to human-readable action names.

        Args:
            grid_idx: (height, grid_size) int64 with values in [0, n_actions]
            action_names: list[str] of action names, indexed 0..n_actions-1

        Returns:
            list[list[str]] of shape (height, grid_size) with action names,
            or "<empty>" for index n_actions
        """
        result = []
        for i in range(self.height):
            row = []
            for j in range(self.grid_size):
                idx = int(grid_idx[i, j].item())
                if idx == self.n_actions:
                    row.append("<empty>")
                else:
                    row.append(action_names[idx])
            result.append(row)
        return result
