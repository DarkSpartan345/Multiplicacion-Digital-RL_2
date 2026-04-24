"""Unit tests for MCTS components."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alphazero.mcts.node import AlphaZeroNode, NodeState
from alphazero.mcts.puct import add_dirichlet_noise, get_policy, puct_scores, select_child


def test_node_expand():
    """Test node expansion with priors."""
    node = AlphaZeroNode()
    priors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    node.expand(priors)
    assert node.expanded, "Node should be marked expanded"
    assert len(node.children) == 4, "Should have 4 children"
    for a in range(4):
        assert node.children[a].P == float(priors[a])
    print("✓ test_node_expand passed")


def test_node_backup():
    """Test value backup through tree."""
    root = AlphaZeroNode()
    root.expand(np.ones(2) * 0.5)
    child = root.children[0]
    child.backup(0.5)
    assert root.N == 1, "Root should have 1 visit"
    assert child.N == 1, "Child should have 1 visit"
    assert abs(root.W - 0.5) < 1e-6, "Root W should be 0.5"
    print("✓ test_node_backup passed")


def test_puct_scores():
    """Test PUCT score computation."""
    root = AlphaZeroNode()
    priors = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    root.expand(priors)
    root.N = 100
    root.children[0].N = 50
    root.children[0].W = 25.0
    root.children[1].N = 30
    root.children[1].W = 15.0
    root.children[2].N = 20
    root.children[2].W = 5.0
    scores = puct_scores(root, c_puct=1.5)
    assert len(scores) == 3
    assert all(isinstance(v, float) for v in scores.values())
    print("✓ test_puct_scores passed")


def test_select_child():
    """Test PUCT-based child selection."""
    root = AlphaZeroNode()
    priors = np.array([0.1, 0.9], dtype=np.float32)
    root.expand(priors)
    root.N = 10
    root.children[0].N = 5
    root.children[0].W = 0.0
    root.children[1].N = 5
    root.children[1].W = 0.0
    child = select_child(root, c_puct=1.5)
    assert isinstance(child, AlphaZeroNode)
    assert child in [root.children[0], root.children[1]]
    print("✓ test_select_child passed")


def test_get_policy():
    """Test policy extraction from node."""
    node = AlphaZeroNode()
    priors = np.ones(4) * 0.25
    node.expand(priors)
    node.N = 100
    node.children[0].N = 50
    node.children[1].N = 30
    node.children[2].N = 15
    node.children[3].N = 5
    policy = get_policy(node, temperature=1.0)
    assert policy.shape == (4,), f"Expected shape (4,), got {policy.shape}"
    assert abs(policy.sum() - 1.0) < 1e-6, "Policy should sum to 1"
    assert abs(policy[0] - 0.5) < 1e-6, "Max child should be 0.5"
    print("✓ test_get_policy passed")


def test_add_dirichlet_noise():
    """Test Dirichlet noise addition."""
    node = AlphaZeroNode()
    priors = np.array([0.5, 0.5], dtype=np.float32)
    node.expand(priors)
    original_priors = [c.P for c in node.children.values()]
    add_dirichlet_noise(node, alpha=1.0, epsilon=0.25)
    new_priors = [c.P for c in node.children.values()]
    for old, new in zip(original_priors, new_priors):
        assert old != new, "Priors should change with noise"
    print("✓ test_add_dirichlet_noise passed")


if __name__ == "__main__":
    test_node_expand()
    test_node_backup()
    test_puct_scores()
    test_select_child()
    test_get_policy()
    test_add_dirichlet_noise()
    print("\n✅ All MCTS tests passed!")
