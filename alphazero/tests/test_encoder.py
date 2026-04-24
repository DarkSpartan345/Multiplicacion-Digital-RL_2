"""Unit tests for StateEncoder."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Environment.env_cuda import BinaryMathEnvCUDA
from alphazero.model.encoder import StateEncoder


def test_encode_empty_grid():
    """Test encoding a completely empty grid."""
    env = BinaryMathEnvCUDA(Bits=4, height=4, n_envs=4, device="cpu")
    env.reset()
    enc = StateEncoder(env)
    g = enc.encode_batch(env.suma_grid)
    assert g.shape == (4, 4, 8), f"Expected (4, 4, 8), got {g.shape}"
    assert g.dtype == torch.int64, f"Expected int64, got {g.dtype}"
    assert (g == enc.n_actions).all(), "All empty cells should map to n_actions"
    print("✓ test_encode_empty_grid passed")


def test_encode_no_minus_one():
    """Test that no -1 values remain after encoding."""
    env = BinaryMathEnvCUDA(Bits=4, height=4, n_envs=1, device="cpu")
    env.reset()
    enc = StateEncoder(env)
    g = enc.encode_batch(env.suma_grid)
    assert (g >= 0).all(), "No negative indices allowed"
    assert (g <= enc.n_actions).all(), f"All indices should be <= {enc.n_actions}"
    print("✓ test_encode_no_minus_one passed")


def test_encode_single():
    """Test single environment encoding."""
    env = BinaryMathEnvCUDA(Bits=4, height=4, n_envs=2, device="cpu")
    env.reset()
    enc = StateEncoder(env)
    g = enc.encode_single(env, 0)
    assert g.shape == (4, 8), f"Expected (4, 8), got {g.shape}"
    assert g.dtype == torch.int64
    print("✓ test_encode_single passed")


if __name__ == "__main__":
    test_encode_empty_grid()
    test_encode_no_minus_one()
    test_encode_single()
    print("\n✅ All encoder tests passed!")
