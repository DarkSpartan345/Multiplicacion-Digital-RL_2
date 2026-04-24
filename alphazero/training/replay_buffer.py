"""Replay buffer for storing self-play experience."""
from collections import deque
from typing import Optional

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing (grid, pi, G_t) tuples.

    All tensors stored on CPU; moved to GPU during batch sampling.
    """

    def __init__(self, maxlen: int = 50000, device: str = "cuda"):
        """Initialize replay buffer.

        Args:
            maxlen: maximum number of samples
            device: 'cuda' or 'cpu' for batch sampling
        """
        self.buffer: deque = deque(maxlen=maxlen)
        self.device = device

    def push(self, grid: Tensor, pi: Tensor, G_t: float) -> None:
        """Store one (state, policy, return) triple.

        Args:
            grid: (height, grid_size) int64
            pi: (n_actions,) float32
            G_t: float return value
        """
        self.buffer.append(
            (
                grid.cpu(),
                pi.float().cpu() if isinstance(pi, Tensor) else torch.tensor(pi).float(),
                torch.tensor(G_t, dtype=torch.float32),
            )
        )

    def push_game(self, samples: list[tuple]) -> None:
        """Batch-push all samples from one game.

        Args:
            samples: list of (grid, pi, G_t) tuples
        """
        for grid, pi, G_t in samples:
            self.push(grid, pi, G_t)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a batch uniformly at random.

        Args:
            batch_size: number of samples to draw

        Returns:
            (grids, pis, returns) all on self.device:
                grids: (batch, height, grid_size) int64
                pis: (batch, n_actions) float32
                returns: (batch, 1) float32
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Buffer has {len(self)} samples but {batch_size} requested"
            )

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        grids = []
        pis = []
        returns = []

        for i in indices:
            grid, pi, G_t = self.buffer[i]
            grids.append(grid)
            pis.append(pi)
            returns.append(G_t)

        grids = torch.stack(grids).to(self.device)
        pis = torch.stack(pis).to(self.device)
        returns = torch.stack(returns).unsqueeze(1).to(self.device)

        return grids, pis, returns

    def __len__(self) -> int:
        """Current buffer size."""
        return len(self.buffer)

    def save(self, path: str) -> None:
        """Save entire buffer to disk.

        Args:
            path: file path to save to
        """
        grids_list = []
        pis_list = []
        returns_list = []

        for grid, pi, G_t in self.buffer:
            grids_list.append(grid)
            pis_list.append(pi)
            returns_list.append(G_t)

        if grids_list:
            checkpoint = {
                "grids": torch.stack(grids_list),
                "pis": torch.stack(pis_list),
                "returns": torch.stack(returns_list),
                "buffer_size": len(self.buffer),
            }
            torch.save(checkpoint, path)
            print(f"[+] Saved buffer with {len(self.buffer)} samples to {path}")

    def load(self, path: str) -> None:
        """Load entire buffer from disk.

        Args:
            path: file path to load from
        """
        checkpoint = torch.load(path, map_location="cpu")
        grids = checkpoint["grids"]
        pis = checkpoint["pis"]
        returns = checkpoint["returns"]

        for i in range(len(grids)):
            self.push(grids[i], pis[i], float(returns[i].item()))

        print(f"[+] Loaded {len(self.buffer)} samples from {path}")
