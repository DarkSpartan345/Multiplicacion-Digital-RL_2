"""Persistent storage for self-play games."""
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class GameStore:
    """Manages saving and loading self-play games to/from disk."""

    def __init__(self, games_dir: str):
        """Initialize game store.

        Args:
            games_dir: directory where to save/load games
        """
        self.games_dir = Path(games_dir)
        self.games_dir.mkdir(parents=True, exist_ok=True)

    def push_game(
        self,
        samples: list[tuple],
        iteration: int,
        final_reward: float = None,
    ) -> str:
        """Save one game to disk.

        Args:
            samples: list of (grid, pi, G_t) tuples
            iteration: iteration number (for naming)
            final_reward: optional, for metadata

        Returns:
            path to saved file
        """
        if not samples:
            return None

        grids = torch.stack([s[0] for s in samples])
        pis_list = []
        for s in samples:
            pi = s[1]
            if isinstance(pi, np.ndarray):
                pi = torch.from_numpy(pi)
            elif isinstance(pi, list):
                pi = torch.tensor(pi)
            elif not isinstance(pi, torch.Tensor):
                pi = torch.tensor(pi)
            pis_list.append(pi)
        pis = torch.stack(pis_list)

        returns_list = []
        for s in samples:
            ret = s[2]
            if isinstance(ret, torch.Tensor):
                returns_list.append(ret)
            else:
                returns_list.append(torch.tensor(ret, dtype=torch.float32))
        returns = torch.stack(returns_list)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{iteration:05d}_{timestamp}.pt"
        filepath = self.games_dir / filename

        checkpoint = {
            "iteration": iteration,
            "timestamp": timestamp,
            "n_steps": len(samples),
            "grids": grids.cpu(),
            "pis": pis.cpu(),
            "returns": returns.cpu(),
            "final_reward": final_reward or 0.0,
        }

        torch.save(checkpoint, filepath)
        return str(filepath)

    def load_all(self, games_dir: str = None) -> list[tuple]:
        """Load all games from directory.

        Args:
            games_dir: directory to load from (uses self.games_dir if None)

        Returns:
            list of (grids, pis, returns) tuples from all saved games
        """
        load_dir = Path(games_dir) if games_dir else self.games_dir

        if not load_dir.exists():
            return []

        game_files = sorted(load_dir.glob("game_*.pt"))
        all_grids = []
        all_pis = []
        all_returns = []
        total_samples = 0

        for filepath in game_files:
            checkpoint = torch.load(filepath, map_location="cpu")
            all_grids.append(checkpoint["grids"])
            all_pis.append(checkpoint["pis"])
            all_returns.append(checkpoint["returns"])
            total_samples += checkpoint["n_steps"]

        if not all_grids:
            return []

        grids = torch.cat(all_grids, dim=0)
        pis = torch.cat(all_pis, dim=0)
        returns = torch.cat(all_returns, dim=0)

        print(
            f"[+] Loaded {len(game_files)} games ({total_samples} samples) "
            f"from {load_dir}"
        )

        return [(grids[i], pis[i].numpy(), float(returns[i].item()))
                for i in range(len(grids))]

    def load_games_to_buffer(self, buffer, games_dir: str = None) -> int:
        """Load all games from directory into a ReplayBuffer.

        Args:
            buffer: ReplayBuffer instance
            games_dir: directory to load from (uses self.games_dir if None)

        Returns:
            number of samples loaded
        """
        samples = self.load_all(games_dir)
        for grid, pi, G_t in samples:
            buffer.push(grid, pi, G_t)
        return len(samples)

    def list_games(self) -> list[dict]:
        """List all saved games with metadata.

        Returns:
            list of dicts with game metadata
        """
        game_files = sorted(self.games_dir.glob("game_*.pt"))
        games = []

        for filepath in game_files:
            checkpoint = torch.load(filepath, map_location="cpu")
            games.append({
                "file": filepath.name,
                "iteration": checkpoint["iteration"],
                "timestamp": checkpoint["timestamp"],
                "n_steps": checkpoint["n_steps"],
                "final_reward": checkpoint["final_reward"],
            })

        return games
