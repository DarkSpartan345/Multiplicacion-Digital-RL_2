#!/usr/bin/env python3
"""Main entry point for AlphaZero binary circuit synthesis."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Environment.env_cuda import BinaryMathEnvCUDA
from alphazero.model.encoder import StateEncoder
from alphazero.model.network import AlphaZeroNet
from alphazero.training.replay_buffer import ReplayBuffer
from alphazero.training.trainer import AlphaZeroTrainer
from alphazero.training.game_store import GameStore
from alphazero.utils.visualization import inspect_buffer


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero MCTS for binary circuit synthesis"
    )
    parser.add_argument("--bits", type=int, default=4, help="Number of operand bits")
    parser.add_argument("--height", type=int, default=4, help="Grid height")
    parser.add_argument(
        "--n-sim", type=int, default=200, help="MCTS simulations per move"
    )
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT constant")
    parser.add_argument(
        "--alpha-blend",
        type=float,
        default=0.5,
        help="Weight for network value vs rollout value",
    )
    parser.add_argument(
        "--n-rollouts", type=int, default=64, help="Parallel rollouts for value"
    )
    parser.add_argument(
        "--games-per-iter", type=int, default=5, help="Self-play games per iteration"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=20,
        help="Training gradient steps per iteration",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument("--d-model", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--n-filters", type=int, default=64, help="CNN filter count")
    parser.add_argument("--n-res", type=int, default=3, help="Number of residual blocks")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use incremental (per-column) rewards",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./alphazero/logs",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./alphazero/checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--games-dir",
        type=str,
        default=None,
        help="Directory to save self-play games",
    )
    parser.add_argument(
        "--load-games",
        type=str,
        default=None,
        help="Directory of saved games to load at startup",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Path to save inspection image (triggers inspection mode)",
    )
    parser.add_argument(
        "--inspect-n",
        type=int,
        default=4,
        help="Number of samples to inspect",
    )

    args = parser.parse_args()

    print(f"[*] Initializing AlphaZero with Bits={args.bits}, height={args.height}")

    play_env = BinaryMathEnvCUDA(
        Bits=args.bits,
        height=args.height,
        n_envs=1,
        device=args.device,
        incremental=args.incremental,
    )
    rollout_env = BinaryMathEnvCUDA(
        Bits=args.bits,
        height=args.height,
        n_envs=max(args.n_rollouts, 64),
        device=args.device,
        incremental=args.incremental,
    )

    encoder = StateEncoder(play_env)
    net = AlphaZeroNet.from_env(
        play_env,
        d_model=args.d_model,
        n_filters=args.n_filters,
        n_res=args.n_res,
    )

    buffer = ReplayBuffer(maxlen=50000, device=args.device)

    trainer = AlphaZeroTrainer(
        net=net,
        encoder=encoder,
        play_env=play_env,
        rollout_env=rollout_env,
        buffer=buffer,
        n_simulations=args.n_sim,
        batch_size=args.batch_size,
        lr=args.lr,
        games_per_iter=args.games_per_iter,
        train_steps_per_iter=args.train_steps,
        alpha_blend=args.alpha_blend,
        n_rollouts=args.n_rollouts,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        games_dir=args.games_dir,
        device=args.device,
    )

    if args.load_games:
        print(f"[*] Loading games from {args.load_games}...")
        gs = GameStore(args.games_dir or "./alphazero/games")
        n_loaded = gs.load_games_to_buffer(buffer, args.load_games)
        print(f"[+] Loaded {n_loaded} samples into buffer")

    if args.inspect:
        print(f"[*] Inspection mode: generating visualizations...")
        fig = inspect_buffer(
            buffer, net, encoder, play_env,
            play_env.possible_actions,
            n_samples=args.inspect_n,
            device=args.device
        )
        fig.savefig(args.inspect, dpi=100)
        print(f"[+] Saved inspection to {args.inspect}")
        return

    start_iteration = 0
    if args.resume:
        start_iteration = trainer.load_checkpoint(args.resume) + 1

    remaining_iterations = args.iterations - start_iteration
    if remaining_iterations > 0:
        trainer.train_loop(remaining_iterations)
    else:
        print("[!] Checkpoint iteration >= requested iterations, skipping training")

    print("[+] Training complete!")


if __name__ == "__main__":
    main()
