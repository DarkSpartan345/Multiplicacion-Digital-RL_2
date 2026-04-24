"""AlphaZero MCTS with neural network guidance."""
import numpy as np
import torch
import torch.nn.functional as F

from alphazero.model.encoder import StateEncoder
from alphazero.model.network import AlphaZeroNet

from .node import AlphaZeroNode, NodeState
from .puct import add_dirichlet_noise, get_policy, select_child


class MCTSSearch:
    """
    AlphaZero-style MCTS with neural network guidance.

    Blends network value estimates with rollout values:
        value = alpha_blend * net_value + (1-alpha_blend) * rollout_value
    """

    def __init__(
        self,
        net: AlphaZeroNet,
        encoder: StateEncoder,
        sim_env,
        rollout_env,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        alpha_blend: float = 0.5,
        n_rollouts: int = 64,
        device: str = "cuda",
    ):
        """Initialize MCTS searcher.

        Args:
            net: AlphaZeroNet
            encoder: StateEncoder
            sim_env: BinaryMathEnvCUDA for simulation state tracking (n_envs >= 1)
            rollout_env: BinaryMathEnvCUDA for rollout evaluation (n_envs >= n_rollouts)
            n_simulations: MCTS simulations per search
            c_puct: PUCT exploration constant
            alpha_blend: weight for net value vs rollout value [0, 1]
            n_rollouts: parallel rollouts for value estimation
            device: 'cuda' or 'cpu'
        """
        self.net = net.eval()
        self.encoder = encoder
        self.sim_env = sim_env
        self.rollout_env = rollout_env
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.alpha_blend = alpha_blend
        self.n_rollouts = min(n_rollouts, rollout_env.n_envs)
        self.device = device

    def search(
        self, root_state: NodeState, add_noise: bool = True
    ) -> tuple[AlphaZeroNode, np.ndarray]:
        """Run n_simulations from root_state.

        Args:
            root_state: NodeState for root of search tree
            add_noise: whether to add Dirichlet noise to root priors

        Returns:
            root: AlphaZeroNode with populated statistics
            policy: (n_actions,) float32 visit-count policy
        """
        root = AlphaZeroNode()
        root.state = root_state

        for _ in range(self.n_simulations):
            self._simulate(root)

        if add_noise and root.expanded:
            add_dirichlet_noise(root)

        policy = get_policy(root, temperature=1.0)
        return root, policy

    def _simulate(self, root: AlphaZeroNode) -> None:
        """Single MCTS simulation: select → expand → evaluate → backup.

        Args:
            root: AlphaZeroNode to start search from
        """
        self._restore_env_state(root.state, self.sim_env, slot=0)
        node = root
        path = [root]

        while node.expanded and not node.is_terminal:
            action = select_child(node, self.c_puct).action
            _, done = self.sim_env.step(
                torch.tensor([action], device=self.sim_env.done.device)
            )

            current_state = NodeState(
                grid=self.sim_env.suma_grid[0].clone().cpu(),
                cursor=int(self.sim_env.cursor_pos[0].cpu().item()),
                reward=float(self.sim_env.rewards[0].cpu().item()),
                is_done=bool(done[0].cpu().item()),
                carry=(
                    self.sim_env.carry_in[0].clone().cpu()
                    if self.sim_env.incremental
                    else None
                ),
            )

            node = select_child(path[-1], self.c_puct)
            node.state = current_state
            path.append(node)

        if node.is_terminal or (node.state and node.state.is_done):
            terminal_reward = node.state.reward if node.state else 0.0
            value = self._normalize_reward(terminal_reward)
        else:
            value = self._evaluate_node(node)

        for n in path:
            n.backup(value)

    def _evaluate_node(self, node: AlphaZeroNode) -> float:
        """Evaluate a leaf node with network + optional rollout.

        Calls network to get policy logits and value.
        Expands all children with softmax priors.
        Optionally blends with rollout value estimate.

        Args:
            node: AlphaZeroNode with state set

        Returns:
            float value in [-1, 1]
        """
        grid_flat = node.state.grid.unsqueeze(0)
        grid_idx = self.encoder.encode_batch(grid_flat).squeeze(0)
        grid_idx = grid_idx.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, net_value = self.net(grid_idx)

        policy_logits = policy_logits.squeeze(0)
        net_value = float(net_value.squeeze().cpu().item())

        priors = F.softmax(policy_logits, dim=-1).cpu().numpy()
        node.expand(priors)

        rollout_value = self._estimate_rollout_value(node.state)

        blended_value = (
            self.alpha_blend * net_value + (1.0 - self.alpha_blend) * rollout_value
        )
        return blended_value

    def _estimate_rollout_value(self, state: NodeState) -> float:
        """Estimate value by rolling out from state.

        Args:
            state: NodeState to start rollout from

        Returns:
            float value in [-1, 1]
        """
        self._restore_env_state(state, self.rollout_env, slot=0)

        state_dict = {
            "suma_grid": state.grid.cpu().numpy().tolist(),
            "cursor_position": state.cursor,
        }
        if state.carry is not None:
            state_dict["carry_in"] = state.carry.cpu().numpy()

        rollout_rewards = self.rollout_env.rollout_from_state(
            state_dict, n_rollouts=self.n_rollouts, rollout_depth=None
        )

        if isinstance(rollout_rewards, torch.Tensor):
            rollout_rewards = rollout_rewards.cpu().numpy()
        rollout_value = float(np.mean(rollout_rewards))
        return self._normalize_reward(rollout_value)

    def _restore_env_state(self, state: NodeState, env, slot: int = 0) -> None:
        """Copy NodeState into env slot without string conversion.

        Args:
            state: NodeState
            env: BinaryMathEnvCUDA
            slot: which env slot to restore to
        """
        env.suma_grid[slot] = state.grid.to(env.suma_grid.device)
        env.cursor_pos[slot] = state.cursor
        env.done[slot] = state.is_done
        env.rewards[slot] = state.reward
        if env.incremental and state.carry is not None:
            env.carry_in[slot] = state.carry.to(env.carry_in.device)

    @staticmethod
    def _normalize_reward(reward: float) -> float:
        """Normalize reward from [-10, 0] to [-1, 1].

        Args:
            reward: float in approximately [-10, 0]

        Returns:
            float in [-1, 1]
        """
        return (reward / 10.0) * 2 + 1
