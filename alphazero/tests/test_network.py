"""Unit tests for AlphaZeroNet."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Environment.env_cuda import BinaryMathEnvCUDA
from alphazero.model.network import AlphaZeroNet


def test_network_output_shapes_bits4():
    """Test network output shapes for Bits=4."""
    net = AlphaZeroNet(n_actions=66, height=4, grid_size=8, d_model=32, n_filters=64, n_res=3)
    net.eval()
    grid = torch.randint(0, 67, (4, 4, 8), dtype=torch.int64)
    with torch.no_grad():
        logits, val = net(grid)
    assert logits.shape == (4, 66), f"Expected (4, 66), got {logits.shape}"
    assert val.shape == (4, 1), f"Expected (4, 1), got {val.shape}"
    print("✓ test_network_output_shapes_bits4 passed")


def test_network_output_shapes_bits8():
    """Test network output shapes for Bits=8."""
    net = AlphaZeroNet(n_actions=258, height=8, grid_size=16, d_model=32, n_filters=64, n_res=3)
    net.eval()
    grid = torch.randint(0, 259, (2, 8, 16), dtype=torch.int64)
    with torch.no_grad():
        logits, val = net(grid)
    assert logits.shape == (2, 258), f"Expected (2, 258), got {logits.shape}"
    assert val.shape == (2, 1), f"Expected (2, 1), got {val.shape}"
    print("✓ test_network_output_shapes_bits8 passed")


def test_value_in_range():
    """Test that value output is in [-1, 1]."""
    net = AlphaZeroNet(n_actions=66, height=4, grid_size=8)
    net.eval()
    grid = torch.randint(0, 67, (32, 4, 8), dtype=torch.int64)
    with torch.no_grad():
        _, val = net(grid)
    assert (val >= -1.0).all() and (val <= 1.0).all(), "Value must be in [-1, 1]"
    print("✓ test_value_in_range passed")


def test_from_env():
    """Test from_env class method."""
    env = BinaryMathEnvCUDA(Bits=4, height=4, n_envs=1, device="cpu")
    net = AlphaZeroNet.from_env(env)
    assert net.n_actions == env.n_actions
    assert net.height == env.height
    assert net.grid_size == env.grid_size
    print("✓ test_from_env passed")


if __name__ == "__main__":
    test_network_output_shapes_bits4()
    test_network_output_shapes_bits8()
    test_value_in_range()
    test_from_env()
    print("\n✅ All network tests passed!")
