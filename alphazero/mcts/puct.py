"""PUCT selection utilities for AlphaZero MCTS."""
import numpy as np

from .node import AlphaZeroNode


def puct_scores(node: AlphaZeroNode, c_puct: float = 1.5) -> dict[int, float]:
    """Compute PUCT score for all children of node.

    Formula: PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Args:
        node: AlphaZeroNode
        c_puct: exploration constant

    Returns:
        dict mapping action -> PUCT score
    """
    scores = {}
    if not node.children:
        return scores

    sqrt_visits = np.sqrt(node.N)
    for action, child in node.children.items():
        q = child.Q
        exploration = c_puct * child.P * sqrt_visits / (1.0 + child.N)
        scores[action] = q + exploration

    return scores


def select_child(node: AlphaZeroNode, c_puct: float = 1.5) -> AlphaZeroNode:
    """Return child with highest PUCT score.

    Args:
        node: AlphaZeroNode
        c_puct: exploration constant

    Returns:
        best child
    """
    scores = puct_scores(node, c_puct)
    if not scores:
        raise ValueError("Node has no children")
    best_action = max(scores, key=scores.get)
    return node.children[best_action]


def add_dirichlet_noise(
    node: AlphaZeroNode, alpha: float = None, epsilon: float = 0.25
) -> None:
    """Add Dirichlet noise to root node's child priors for exploration.

    Args:
        node: root AlphaZeroNode (should not have parent)
        alpha: Dirichlet parameter (default: 10.0 / n_actions)
        epsilon: mixing parameter (default: 0.25)
    """
    if not node.children:
        raise ValueError("Node must be expanded before adding noise")

    n_actions = len(node.children)
    if alpha is None:
        alpha = 10.0 / n_actions

    noise = np.random.dirichlet([alpha] * n_actions)

    for a, child in node.children.items():
        child.P = (1.0 - epsilon) * child.P + epsilon * noise[a]


def get_policy(node: AlphaZeroNode, temperature: float = 1.0) -> np.ndarray:
    """Convert visit counts to policy vector for self-play data generation.

    Args:
        node: AlphaZeroNode (should be root of a completed search)
        temperature: exploration temperature
            - 1.0: pi_a = N(a) / sum_b N(b)
            - 0.0: one-hot at argmax N

    Returns:
        (n_actions,) float32 policy summing to 1.0
    """
    counts = node.visit_counts()
    if len(counts) == 0 or counts.sum() == 0:
        raise ValueError("Node has no visits")

    if temperature == 0.0:
        policy = np.zeros_like(counts, dtype=np.float32)
        policy[np.argmax(counts)] = 1.0
        return policy

    counts_temp = np.power(counts.astype(np.float32), 1.0 / temperature)
    policy = counts_temp / counts_temp.sum()
    return policy.astype(np.float32)
