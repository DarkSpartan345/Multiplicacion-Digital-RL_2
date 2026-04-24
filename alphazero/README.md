# AlphaZero MCTS for Binary Circuit Synthesis

A complete implementation of AlphaZero-style MCTS with neural network policy and value heads for learning to synthesize binary multiplier circuits.

## Quick Start

```bash
# Bits=4 (66 actions), 200 MCTS sims per move, 100 training iterations
python3 alphazero/main.py --bits 4 --height 4 --n-sim 200 --iterations 100

# Bits=8 (258 actions), fewer sims, blended value, more rollouts
python3 alphazero/main.py --bits 8 --height 8 --n-sim 100 --iterations 50 --alpha-blend 0.7 --n-rollouts 128

# Resume from checkpoint
python3 alphazero/main.py --bits 4 --resume alphazero/checkpoints/checkpoint_00009.pt
```

## Architecture

### 1. State Encoding (`model/encoder.py`)
- Converts env grid `(n_envs, CC)` int16 with -1 (empty) into `(n_envs, height, grid_size)` int64
- Empty cells mapped to index `n_actions` for Embedding lookup
- Single int64 mapping avoids string-based conversion overhead

### 2. Neural Network (`model/network.py`)
- **Embedding**: `(n_actions+1, d_model=32)` learns representation of each grid cell
- **Stem**: Conv2d + BN + ReLU converts spatial features
- **Body**: 3× ResBlock with skip connections (preserves spatial structure)
- **Policy Head**: 1×1 Conv → Flatten → Linear → (n_actions,) logits
- **Value Head**: 1×1 Conv → Flatten → FC → FC → Tanh → (1,) in [-1, 1]

### 3. MCTS Nodes (`mcts/node.py`)
- **NodeState**: dataclass storing grid, cursor, reward, is_done, carry
- **AlphaZeroNode**: stores N (visits), W (value sum), P (prior), children dict
- **Q property**: W/N (mean value)
- **expand()**: creates all n_actions children with network priors
- **backup()**: single-agent propagation (no sign flipping)

### 4. PUCT Selection (`mcts/puct.py`)
- **puct_scores()**: `Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- **select_child()**: argmax PUCT score
- **add_dirichlet_noise()**: exploration at root (only in first move)
- **get_policy()**: extract visit-count policy from node

### 5. MCTS Search (`mcts/search.py`)
- **MCTSSearch.search()**: runs n_simulations from root_state
- **_simulate()**: selection → expansion → evaluation → backup
- **_evaluate_node()**: calls network, expands children, blends value
- **Value blending**: `value = α * net_value + (1-α) * rollout_value`
- **Reward normalization**: `(r/10)*2+1` maps [-10, 0] → [-1, 1]

Key design: uses separate `sim_env` (n_envs=1) to track tree descent and `rollout_env` (n_envs=64+) for parallel value estimation. This prevents state corruption between simulations.

### 6. Self-Play (`training/self_play.py`)
- **SelfPlayWorker.play_game()**: executes one full game
  1. Reset env
  2. Loop until done:
     - Run MCTS to get root, policy
     - Sample action from visit counts
     - Advance env
     - Encode state, save (grid, pi, step)
  3. Compute returns G_t with discounting (gamma=0.99)
  4. Normalize to [-1, 1]
  5. Return [(grid, pi, G_t), ...]

### 7. Training Loop (`training/trainer.py`)
- **AlphaZeroTrainer.train_loop()**: main training loop
  1. play_games() → collect samples
  2. buffer.push_game() → store in replay buffer
  3. if buffer >= batch_size: train for train_steps_per_iter:
     - sample batch
     - forward net
     - policy_loss = -sum(pi * log_softmax(logits))
     - value_loss = MSE(pred, target)
     - backward, optimize
  4. Log metrics every iteration
  5. Save checkpoint every 10 iterations

### 8. Metrics Logging (`utils/metrics.py`)
- TensorBoard scalars:
  - `Training/total_loss`, `policy_loss`, `value_loss`
  - `Training/policy_entropy` (max ≈ ln(n_actions))
  - `SelfPlay/final_reward`, `trajectory_length`
  - `Buffer/size`

### 9. Replay Buffer (`training/replay_buffer.py`)
- Circular deque with maxlen=50000
- Stores (grid, pi, G_t) on CPU
- Samples to GPU in batches

## Command-Line Arguments

```
--bits INT               Operand bit width (default 4)
--height INT             Grid height (default 4)
--n-sim INT              MCTS simulations per move (default 200)
--c-puct FLOAT           PUCT exploration constant (default 1.5)
--alpha-blend FLOAT      Network vs rollout value blend (default 0.5)
--n-rollouts INT         Parallel rollouts for value (default 64)
--games-per-iter INT     Self-play games per iteration (default 5)
--train-steps INT        Training steps per iteration (default 20)
--batch-size INT         Training batch size (default 256)
--lr FLOAT               Adam learning rate (default 1e-3)
--iterations INT         Total training iterations (default 100)
--d-model INT            Embedding dimension (default 32)
--n-filters INT          CNN filter count (default 64)
--n-res INT              Residual blocks (default 3)
--incremental            Use per-column rewards instead of terminal
--log-dir STR            TensorBoard log directory
--checkpoint-dir STR     Checkpoint save directory
--resume STR             Path to checkpoint to resume from
--device cuda|cpu        Computation device (default cuda)
```

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Value blending | `α*net + (1-α)*rollout` | Cold-start: untrained net ≈ random, rollouts provide real signal from iteration 1 |
| No sign flipping | Single-agent backup | Unlike two-player games; value represents circuit quality, not alternating perspectives |
| Two-env pattern | Separate sim_env + rollout_env | Avoids state corruption; `rollout_from_state` resets slots |
| All children at once | Immediate expansion | AlphaZero standard; PUCT explores based on priors; unvisited children have high exploration term |
| Reward normalization | `(r/10)*2+1` maps [-10,0]→[-1,1] | Network Tanh output in [-1,1]; targets must match |
| Incremental mode | Off by default | Denser rewards help value learning, but may bias away from global optima; terminal reward cleaner |

## Tests

```bash
# Encoder
python3 alphazero/tests/test_encoder.py

# Network
python3 alphazero/tests/test_network.py

# MCTS
python3 alphazero/tests/test_mcts.py

# All together (Bits=2, 20 sims, 1 iteration)
python3 alphazero/main.py --bits 2 --n-sim 20 --iterations 1 --games-per-iter 1
```

## Expected Behavior

**Iteration 1-5:** Network outputs random policies, rollouts guide MCTS. Loss high (~4-5). Reward ≈ -8 (mostly bad circuits).

**Iteration 10-20:** Network policy improves slightly, value head starts correlating. Loss decreases (~2-3). Some circuits slightly better.

**Iteration 50+:** Network value becomes reliable, can phase out rollouts (increase alpha_blend gradually). Loss converges (~0.5-1). Some circuits reach good solutions.

## Performance Estimates

| Bits | Height | n_actions | Tree Nodes (200 sims) | Per-Move Time | Per-Game Time |
|------|--------|-----------|----------------------|---------------|---------------|
| 2    | 2      | 18        | ~200                 | 50 ms         | 300 ms        |
| 4    | 4      | 66        | ~200                 | 80 ms         | 2.5 s         |
| 8    | 8      | 258       | ~200                 | 100 ms        | 13 s          |

Times include network eval (7 ms), rollout (18 ms), env step (1 ms), Python overhead.

## Monitoring Training

```bash
tensorboard --logdir alphazero/logs
```

Watch for:
- `total_loss` decreasing over time (especially after iteration 10)
- `policy_entropy` starting high (~3-4) and remaining stable
- `final_reward` slowly improving (toward -0, ideally)
- Checkpoints appearing every 10 iterations

## Future Improvements

1. **Alpha_blend scheduling**: linearly increase from 0.5 → 0.9 over 100 iters
2. **Virtual loss** in MCTS: batch expand multiple leaves per simulation
3. **Lazy child expansion**: only create children when first visited (saves memory for Bits=8)
4. **Intrinsic reward shaping**: bonus for column correctness (incremental mode)
5. **Curriculum learning**: start with Bits=2, transfer weights to Bits=4, etc.
