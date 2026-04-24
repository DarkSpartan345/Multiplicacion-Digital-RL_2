"""Visualization utilities for inspecting states, policies, and model predictions."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch import Tensor


def plot_grid(grid_idx: Tensor, action_names: list[str], title: str = "Grid State") -> plt.Figure:
    """Visualize a grid state as a colored table.

    Args:
        grid_idx: (height, grid_size) int64 tensor with indices [0, n_actions]
        action_names: list of action names indexed 0..n_actions-1
        title: subplot title

    Returns:
        matplotlib Figure
    """
    grid = grid_idx.cpu().numpy() if isinstance(grid_idx, Tensor) else grid_idx
    height, grid_size = grid.shape
    n_actions = len(action_names)

    fig, ax = plt.subplots(figsize=(grid_size * 0.5, height * 0.5))

    color_map = plt.cm.get_cmap("tab20", n_actions + 1)
    grid_colored = np.zeros((height, grid_size, 3))
    for i in range(height):
        for j in range(grid_size):
            idx = int(grid[i, j])
            color = color_map(idx / (n_actions + 1))[:3]
            grid_colored[i, j] = color

    ax.imshow(grid_colored)

    for i in range(height):
        for j in range(grid_size):
            idx = int(grid[i, j])
            if idx == n_actions:
                text = "∅"
            else:
                text = action_names[idx][:3]
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="white", weight="bold")

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(height))
    ax.set_xticklabels(range(grid_size), fontsize=8)
    ax.set_yticklabels(range(height), fontsize=8)
    ax.set_xlabel("Column (bit position)", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(True, alpha=0.3, color="gray", linestyle="--")

    plt.tight_layout()
    return fig


def plot_policy(policy: np.ndarray, action_names: list[str], top_k: int = 15,
                title: str = "Policy Distribution") -> plt.Figure:
    """Visualize action policy as a bar chart.

    Args:
        policy: (n_actions,) float32 array summing to 1.0
        action_names: list of action names
        top_k: show only top-K actions
        title: subplot title

    Returns:
        matplotlib Figure
    """
    policy = policy.cpu().numpy() if isinstance(policy, Tensor) else policy

    top_indices = np.argsort(policy)[-top_k:][::-1]
    top_probs = policy[top_indices]
    top_names = [action_names[i] if i < len(action_names) else f"Empty({i})" for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
    bars = ax.barh(range(len(top_indices)), top_probs, color=colors)

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.invert_yaxis()

    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(prob, i, f" {prob:.3f}", va="center", fontsize=8)

    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig


def plot_metadata(G_t: float, step: int, final_reward: float = None,
                  value_pred: float = None, entropy: float = None) -> plt.Figure:
    """Visualize metadata about a state.

    Args:
        G_t: discounted return
        step: step number in episode
        final_reward: final reward of episode (if available)
        value_pred: predicted value from network (if available)
        entropy: policy entropy (if available)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    text_lines = [
        f"Return (G_t): {G_t:.4f}",
        f"Step: {step}",
    ]
    if final_reward is not None:
        text_lines.append(f"Final Reward: {final_reward:.4f}")
    if value_pred is not None:
        text_lines.append(f"Value Pred: {value_pred:.4f}")
    if entropy is not None:
        text_lines.append(f"Entropy: {entropy:.4f}")

    text_str = "\n".join(text_lines)
    ax.text(0.5, 0.5, text_str, ha="center", va="center", fontsize=12,
            family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    return fig


def inspect_sample(grid: Tensor, pi: np.ndarray, G_t: float, net,
                   encoder, env, action_names: list[str],
                   step: int = 0, final_reward: float = None) -> plt.Figure:
    """Create a 2×2 subplot inspection of a single sample.

    Args:
        grid: (height, grid_size) int64 encoded state
        pi: (n_actions,) float32 policy
        G_t: discounted return
        net: AlphaZeroNet (will be called to get value prediction)
        encoder: StateEncoder
        env: BinaryMathEnvCUDA (for dimensions)
        action_names: list of action names
        step: step number in episode
        final_reward: final reward (optional)

    Returns:
        matplotlib Figure with 2×2 subplots
    """
    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    grid_np = grid.cpu().numpy() if isinstance(grid, Tensor) else grid
    pi_np = pi.cpu().numpy() if isinstance(pi, Tensor) else pi

    grid_colored = np.zeros((env.height, env.grid_size, 3))
    color_map = plt.cm.get_cmap("tab20", env.n_actions + 1)
    for i in range(env.height):
        for j in range(env.grid_size):
            idx = int(grid_np[i, j])
            color = color_map(idx / (env.n_actions + 1))[:3]
            grid_colored[i, j] = color

    ax1.imshow(grid_colored)
    for i in range(env.height):
        for j in range(env.grid_size):
            idx = int(grid_np[i, j])
            text = "∅" if idx == env.n_actions else action_names[idx][:2]
            ax1.text(j, i, text, ha="center", va="center", fontsize=7, color="white", weight="bold")
    ax1.set_title("Grid State", fontsize=10, weight="bold")
    ax1.set_xticks(range(env.grid_size))
    ax1.set_yticks(range(env.height))
    ax1.grid(True, alpha=0.3)

    top_k = min(10, env.n_actions)
    top_indices = np.argsort(pi_np)[-top_k:][::-1]
    top_probs = pi_np[top_indices]
    top_names = [action_names[i] if i < len(action_names) else f"a{i}" for i in top_indices]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
    ax2.barh(range(len(top_indices)), top_probs, color=colors)
    ax2.set_yticks(range(len(top_indices)))
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.set_xlabel("Probability", fontsize=9)
    ax2.set_title("Top-10 Actions", fontsize=10, weight="bold")
    ax2.invert_yaxis()
    for i, prob in enumerate(top_probs):
        ax2.text(prob, i, f" {prob:.3f}", va="center", fontsize=7)

    grid_tensor = grid.unsqueeze(0).to(next(net.parameters()).device)
    with torch.no_grad():
        _, value_pred = net(grid_tensor)
        value_pred = float(value_pred.squeeze().cpu().item())

    text_lines = [
        f"Return (G_t): {G_t:.4f}",
        f"Step: {step}",
        f"Value Pred: {value_pred:.4f}",
    ]
    if final_reward is not None:
        text_lines.append(f"Final Reward: {final_reward:.4f}")

    ax3.axis("off")
    text_str = "\n".join(text_lines)
    ax3.text(0.5, 0.5, text_str, ha="center", va="center", fontsize=11,
             family="monospace", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    ax3.set_title("Metadata", fontsize=10, weight="bold")

    ax4.hist(pi_np, bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Policy Probability", fontsize=9)
    ax4.set_ylabel("Frequency", fontsize=9)
    ax4.set_title("Policy Distribution", fontsize=10, weight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def inspect_buffer(buffer, net, encoder, env, action_names: list[str],
                   n_samples: int = 4, device: str = "cuda") -> plt.Figure:
    """Create a grid of subplots inspecting N random samples from buffer.

    Args:
        buffer: ReplayBuffer
        net: AlphaZeroNet
        encoder: StateEncoder
        env: BinaryMathEnvCUDA
        action_names: list of action names
        n_samples: number of samples to inspect
        device: computation device

    Returns:
        matplotlib Figure with grid of subplots
    """
    if len(buffer) < n_samples:
        print(f"⚠️  Buffer has {len(buffer)} samples, inspecting all.")
        n_samples = len(buffer)

    grids, pis, returns = buffer.sample(n_samples)
    grids = grids.cpu()
    pis = pis.cpu()
    returns = returns.cpu()

    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, 5 * n_rows))

    for idx in range(n_samples):
        grid = grids[idx]
        pi = pis[idx].numpy()
        G_t = float(returns[idx, 0].item())

        ax_grid = plt.subplot(n_rows, n_cols, idx + 1)

        grid_np = grid.cpu().numpy() if isinstance(grid, Tensor) else grid
        grid_colored = np.zeros((env.height, env.grid_size, 3))
        color_map = plt.cm.get_cmap("tab20", env.n_actions + 1)
        for i in range(env.height):
            for j in range(env.grid_size):
                idx_val = int(grid_np[i, j])
                color = color_map(idx_val / (env.n_actions + 1))[:3]
                grid_colored[i, j] = color

        ax_grid.imshow(grid_colored)
        for i in range(env.height):
            for j in range(env.grid_size):
                idx_val = int(grid_np[i, j])
                text = "∅" if idx_val == env.n_actions else action_names[idx_val][:2]
                ax_grid.text(j, i, text, ha="center", va="center", fontsize=6, color="white")

        top_action = int(np.argmax(pi[:env.n_actions]))
        top_action_name = action_names[top_action] if top_action < len(action_names) else f"a{top_action}"
        ax_grid.set_title(f"Sample {idx+1} | G_t={G_t:.3f} | Top: {top_action_name}", fontsize=9)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

    plt.tight_layout()
    return fig
