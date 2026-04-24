"""Self-play worker for generating AlphaZero training data."""
import numpy as np
import torch
from torch import Tensor

from alphazero.model.encoder import StateEncoder
from alphazero.mcts.node import NodeState
from alphazero.mcts.puct import get_policy
from alphazero.mcts.search import MCTSSearch


class SelfPlayWorker:
    """
    Generates self-play games and collects training data.

    Returns per game:
        List of (grid_idx, pi, G_t) tuples where:
            grid_idx: (height, grid_size) int64 — encoded state
            pi: (n_actions,) float32 — MCTS visit-count policy
            G_t: float — discounted return from step t to end
    """

    def __init__(
        self,
        net,
        encoder: StateEncoder,
        play_env,
        rollout_env,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        alpha_blend: float = 0.5,
        n_rollouts: int = 64,
        gamma: float = 0.99,
        temperature_threshold: float = 0.33,
        device: str = "cuda",
    ):
        """Initialize self-play worker.

        Args:
            net: AlphaZeroNet
            encoder: StateEncoder
            play_env: BinaryMathEnvCUDA for game progression (n_envs >= 1)
            rollout_env: BinaryMathEnvCUDA for MCTS rollout (n_envs >= n_rollouts)
            n_simulations: MCTS simulations per move
            c_puct: PUCT exploration constant
            alpha_blend: network vs rollout value blend
            n_rollouts: parallel rollouts per MCTS evaluation
            gamma: reward discounting factor
            temperature_threshold: fraction of game where temperature=1.0
            device: 'cuda' or 'cpu'
        """
        self.net = net
        self.encoder = encoder
        self.play_env = play_env
        self.rollout_env = rollout_env
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.alpha_blend = alpha_blend
        self.n_rollouts = n_rollouts
        self.gamma = gamma
        self.temperature_threshold = temperature_threshold
        self.device = device

    def play_game(self) -> list[tuple[Tensor, np.ndarray, float]]:
        """Run one complete game and return training samples.

        Returns:
            List of (grid_idx, pi, G_t) tuples
        """
        self.play_env.reset()
        trajectory = []
        rewards_per_step = []

        for step in range(self.play_env.CC):
            root_state = self._get_root_state(0)

            mcts = MCTSSearch(
                self.net,
                self.encoder,
                self.play_env,
                self.rollout_env,
                n_simulations=self.n_simulations,
                c_puct=self.c_puct,
                alpha_blend=self.alpha_blend,
                n_rollouts=self.n_rollouts,
                device=self.device,
            )

            root, policy = mcts.search(root_state, add_noise=(step == 0))

            temperature = (
                1.0 if step < self.play_env.CC * self.temperature_threshold else 0.0
            )
            action = self._sample_action(root, temperature)

            grid_idx = self.encoder.encode_single(self.play_env, 0)
            trajectory.append(
                {
                    "grid": grid_idx,
                    "pi": policy,
                    "step": step,
                }
            )

            _, done = self.play_env.step(torch.tensor([action], device=self.play_env.rewards.device))
            rewards_per_step.append(float(self.play_env.rewards[0].cpu().item()))

            if bool(done[0].cpu().item()):
                break

        samples = []
        T = len(trajectory)
        for t in range(T):
            if self.gamma < 1.0:
                G_t = sum(
                    self.gamma ** (k - t) * rewards_per_step[k]
                    for k in range(t, T)
                )
            else:
                G_t = rewards_per_step[-1]

            G_t = MCTSSearch._normalize_reward(G_t)

            samples.append((trajectory[t]["grid"], trajectory[t]["pi"], G_t))

        return samples

    def play_games(self, n_games: int) -> list[tuple]:
        """Run n_games sequentially and aggregate samples.

        Args:
            n_games: number of games to play

        Returns:
            Flat list of all (grid, pi, G_t) samples
        """
        all_samples = []
        for _ in range(n_games):
            samples = self.play_game()
            all_samples.extend(samples)
        return all_samples

    def _get_root_state(self, env_idx: int) -> NodeState:
        """Extract current env state as NodeState.

        Args:
            env_idx: which env slot

        Returns:
            NodeState for MCTS root
        """
        return NodeState(
            grid=self.play_env.suma_grid[env_idx].clone().cpu(),
            cursor=int(self.play_env.cursor_pos[env_idx].cpu().item()),
            reward=float(self.play_env.rewards[env_idx].cpu().item()),
            is_done=bool(self.play_env.done[env_idx].cpu().item()),
            carry=(
                self.play_env.carry_in[env_idx].clone().cpu()
                if self.play_env.incremental
                else None
            ),
        )

    @staticmethod
    def _sample_action(root, temperature: float) -> int:
        """Sample action from MCTS root visit counts.

        Args:
            root: AlphaZeroNode (root of completed search)
            temperature: exploration temperature (0=argmax, 1=softmax)

        Returns:
            int action index
        """
        pi = get_policy(root, temperature=temperature)
        return np.random.choice(len(pi), p=pi)
