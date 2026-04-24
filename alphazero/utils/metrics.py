"""TensorBoard metrics logging for AlphaZero."""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """Structured TensorBoard logging for AlphaZero training."""

    def __init__(self, log_dir: str = "./alphazero/logs"):
        """Initialize metrics logger.

        Args:
            log_dir: directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_train_step(
        self, total_loss: float, policy_loss: float, value_loss: float, policy_logits
    ) -> None:
        """Log metrics after each gradient step.

        Args:
            total_loss: scalar
            policy_loss: scalar
            value_loss: scalar
            policy_logits: (batch, n_actions) Tensor for entropy computation
        """
        self.writer.add_scalar("Training/total_loss", total_loss, self.step)
        self.writer.add_scalar("Training/policy_loss", policy_loss, self.step)
        self.writer.add_scalar("Training/value_loss", value_loss, self.step)

        probs = F.softmax(policy_logits.detach(), dim=-1)
        log_probs = F.log_softmax(policy_logits.detach(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        self.writer.add_scalar(
            "Training/policy_entropy", entropy.item(), self.step
        )

        self.step += 1

    def log_self_play_game(
        self, final_reward: float, trajectory_length: int
    ) -> None:
        """Log metrics after each self-play game.

        Args:
            final_reward: terminal reward from game
            trajectory_length: number of steps in game
        """
        self.writer.add_scalar("SelfPlay/final_reward", final_reward, self.step)
        self.writer.add_scalar("SelfPlay/trajectory_length", trajectory_length, self.step)

    def log_iteration(self, iteration: int, buffer_size: int) -> None:
        """Log metrics at end of iteration.

        Args:
            iteration: iteration number
            buffer_size: current replay buffer size
        """
        self.writer.add_scalar("Buffer/size", buffer_size, iteration)

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()
