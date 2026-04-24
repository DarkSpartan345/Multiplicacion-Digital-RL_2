"""MCTS node classes for AlphaZero."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class NodeState:
    """Compact snapshot of environment state for a single MCTS node."""

    grid: Tensor
    cursor: int
    reward: float
    is_done: bool
    carry: Optional[Tensor] = None

    def to_device(self, device: str) -> "NodeState":
        """Move tensors to device."""
        return NodeState(
            grid=self.grid.to(device),
            cursor=self.cursor,
            reward=self.reward,
            is_done=self.is_done,
            carry=self.carry.to(device) if self.carry is not None else None,
        )


class AlphaZeroNode:
    """
    MCTS node with neural network priors.

    Statistics:
        N: visit count
        W: accumulated value from backprop
        P: prior from network softmax
    """

    __slots__ = [
        "parent",
        "action",
        "N",
        "W",
        "P",
        "children",
        "state",
        "expanded",
        "is_terminal",
    ]

    def __init__(
        self,
        parent: Optional["AlphaZeroNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        """Initialize node.

        Args:
            parent: parent AlphaZeroNode or None for root
            action: action that led to this node, or None for root
            prior: prior probability from network softmax
        """
        self.parent: Optional[AlphaZeroNode] = parent
        self.action: Optional[int] = action
        self.N: int = 0
        self.W: float = 0.0
        self.P: float = prior
        self.children: dict[int, "AlphaZeroNode"] = {}
        self.state: Optional[NodeState] = None
        self.expanded: bool = False
        self.is_terminal: bool = False

    @property
    def Q(self) -> float:
        """Mean value of this node."""
        return self.W / self.N if self.N > 0 else 0.0

    def expand(self, priors: np.ndarray) -> None:
        """Create all n_actions children with their priors.

        Args:
            priors: (n_actions,) float32, softmax-normalized.
        """
        for a, p in enumerate(priors):
            self.children[a] = AlphaZeroNode(parent=self, action=a, prior=float(p))
        self.expanded = True

    def backup(self, value: float) -> None:
        """Propagate value up the tree without sign flipping (single-agent).

        Args:
            value: float in [-1, 1] (or normalized reward range)
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

    def visit_counts(self) -> np.ndarray:
        """Return visit counts for all children.

        Returns:
            (n_actions,) int32 array where index a contains N of child a,
            or 0 if child a does not exist.
        """
        if not self.children:
            return np.zeros(0, dtype=np.int32)
        max_action = max(self.children.keys())
        counts = np.zeros(max_action + 1, dtype=np.int32)
        for a, child in self.children.items():
            counts[a] = child.N
        return counts
