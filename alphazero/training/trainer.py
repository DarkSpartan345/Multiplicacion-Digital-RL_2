"""AlphaZero training loop."""
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from alphazero.model.encoder import StateEncoder
from alphazero.model.network import AlphaZeroNet
from alphazero.utils.metrics import MetricsLogger

from .replay_buffer import ReplayBuffer
from .self_play import SelfPlayWorker
from .game_store import GameStore


class AlphaZeroTrainer:
    """Manages AlphaZero training: self-play, training, checkpointing."""

    def __init__(
        self,
        net: AlphaZeroNet,
        encoder: StateEncoder,
        play_env,
        rollout_env,
        buffer: ReplayBuffer,
        n_simulations: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        games_per_iter: int = 5,
        train_steps_per_iter: int = 20,
        alpha_blend: float = 0.5,
        n_rollouts: int = 64,
        log_dir: str = "./alphazero/logs",
        checkpoint_dir: str = "./alphazero/checkpoints",
        games_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        """Initialize trainer.

        Args:
            net: AlphaZeroNet
            encoder: StateEncoder
            play_env: BinaryMathEnvCUDA for self-play
            rollout_env: BinaryMathEnvCUDA for MCTS rollout
            buffer: ReplayBuffer
            n_simulations: MCTS simulations per move
            batch_size: training batch size
            lr: learning rate
            weight_decay: L2 regularization
            games_per_iter: self-play games per iteration
            train_steps_per_iter: gradient steps per iteration
            alpha_blend: network vs rollout value blend
            n_rollouts: parallel rollouts per MCTS
            log_dir: directory for TensorBoard logs
            checkpoint_dir: directory for model checkpoints
            games_dir: directory for saving games (optional)
            device: 'cuda' or 'cpu'
        """
        self.net = net.to(device)
        self.encoder = encoder
        self.play_env = play_env
        self.rollout_env = rollout_env
        self.buffer = buffer
        self.n_simulations = n_simulations
        self.batch_size = batch_size
        self.games_per_iter = games_per_iter
        self.train_steps_per_iter = train_steps_per_iter
        self.alpha_blend = alpha_blend
        self.n_rollouts = n_rollouts
        self.device = device

        self.optimizer = optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.metrics = MetricsLogger(log_dir)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.game_store: Optional[GameStore] = None
        if games_dir:
            self.game_store = GameStore(games_dir)

        self.worker = SelfPlayWorker(
            net=net,
            encoder=encoder,
            play_env=play_env,
            rollout_env=rollout_env,
            n_simulations=n_simulations,
            alpha_blend=alpha_blend,
            n_rollouts=n_rollouts,
            device=device,
        )

    def train_loop(self, n_iterations: int) -> None:
        """Main training loop.

        Args:
            n_iterations: number of training iterations
        """
        for iteration in range(n_iterations):
            if self.n_simulations > 0:
                samples = self.worker.play_games(self.games_per_iter)
                self.buffer.push_game(samples)

                if self.game_store:
                    final_reward = samples[-1][2] if samples else 0.0
                    self.game_store.push_game(samples, iteration, final_reward)

            if len(self.buffer) >= self.batch_size:
                for step in range(self.train_steps_per_iter):
                    grids, target_pi, target_G = self.buffer.sample(self.batch_size)

                    total_loss, policy_loss, value_loss = self.compute_loss(
                        grids, target_pi, target_G
                    )

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    self.metrics.log_train_step(
                        total_loss.item(),
                        policy_loss.item(),
                        value_loss.item(),
                        self.net(grids)[0],
                    )

            mode = "producer+training" if self.n_simulations > 0 else "training-only"
            print(
                f"Iteration {iteration+1}/{n_iterations} [{mode}]: "
                f"buffer_size={len(self.buffer)}"
            )

            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(iteration)

        self.metrics.close()

    def compute_loss(self, grids, target_pi, target_G):
        """Compute policy and value losses.

        Args:
            grids: (batch, height, grid_size) int64
            target_pi: (batch, n_actions) float32
            target_G: (batch, 1) float32

        Returns:
            (total_loss, policy_loss, value_loss) all scalar Tensors
        """
        self.net.train()

        policy_logits, value_pred = self.net(grids)

        policy_loss = -(target_pi * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value_pred, target_G)
        total_loss = policy_loss + value_loss

        self.net.eval()

        return total_loss, policy_loss, value_loss

    def save_checkpoint(self, iteration: int) -> None:
        """Save checkpoint.

        Args:
            iteration: current iteration number
        """
        checkpoint = {
            "iteration": iteration,
            "net_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:05d}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint.

        Args:
            path: path to checkpoint file

        Returns:
            iteration number from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        iteration = checkpoint["iteration"]
        print(f"Loaded checkpoint from {path}, iteration {iteration}")
        return iteration
