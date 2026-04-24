"""AlphaZero network: CNN with policy and value heads."""
import torch
import torch.nn as nn
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block: Conv → BN → ReLU → Conv → BN with skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero network: Embedding → CNN body → policy + value heads.

    Args:
        n_actions (int): number of valid actions
        height (int): grid rows
        grid_size (int): grid columns = 2*Bits
        d_model (int): embedding dimension (default 32)
        n_filters (int): CNN filter count (default 64)
        n_res (int): number of residual blocks (default 3)
    """

    def __init__(
        self,
        n_actions: int,
        height: int,
        grid_size: int,
        d_model: int = 32,
        n_filters: int = 64,
        n_res: int = 3,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.height = height
        self.grid_size = grid_size
        self.d_model = d_model
        self.n_filters = n_filters

        self.embedding = nn.Embedding(n_actions + 1, d_model)
        self.stem = nn.Sequential(
            nn.Conv2d(d_model, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(n_filters) for _ in range(n_res)])

        spatial_size = height * grid_size

        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * spatial_size, n_actions)

        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(spatial_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, grid_idx: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            grid_idx: (batch, height, grid_size) int64 with values in [0, n_actions]

        Returns:
            policy_logits: (batch, n_actions) — raw logits, NOT softmax
            value: (batch, 1) — in [-1, 1] via Tanh
        """
        batch_size = grid_idx.shape[0]

        x = self.embedding(grid_idx)
        x = x.permute(0, 3, 1, 2)

        x = self.stem(x)
        x = self.body(x)

        policy = self.policy_head(x)
        policy = policy.reshape(batch_size, -1)
        policy_logits = self.policy_fc(policy)

        value = self.value_head(x)
        value = value.reshape(batch_size, -1)
        value = self.value_fc1(value)
        value = torch.relu(value)
        value = self.value_fc2(value)
        value = self.value_tanh(value)

        return policy_logits, value

    @classmethod
    def from_env(cls, env, **kwargs) -> "AlphaZeroNet":
        """Construct network from environment dimensions.

        Args:
            env: BinaryMathEnvCUDA instance
            **kwargs: additional arguments for __init__

        Returns:
            AlphaZeroNet instance
        """
        return cls(env.n_actions, env.height, env.grid_size, **kwargs)
