# 🎮 AlphaZero: Game Persistence & Visualization Guide

Complete guide to training with persistent games and inspecting what your network learns.

---

## 🚀 Quick Start: Multi-Session Training

### Session 1: Start training and save games
```bash
python3 alphazero/main.py \
  --bits 4 \
  --height 4 \
  --n-sim 150 \
  --iterations 500 \
  --games-per-iter 8 \
  --train-steps 30 \
  --batch-size 256 \
  --games-dir ./games_session1
```

**What happens:**
- Runs 500 iterations (≈4-6 hours on GTX 1070 Ti)
- Saves every game to `./games_session1/game_*.pt`
- Saves checkpoints every 10 iterations to `./checkpoints/`
- Logs TensorBoard events to `./logs/`

### Session 2: Load previous games and continue
```bash
python3 alphazero/main.py \
  --bits 4 \
  --height 4 \
  --n-sim 150 \
  --iterations 500 \
  --games-per-iter 8 \
  --load-games ./games_session1 \
  --resume ./checkpoints/checkpoint_00499.pt \
  --games-dir ./games_session2
```

**What happens:**
- Loads all 500×8 = 4000 games from Session 1 into ReplayBuffer
- Resumes from iteration 500 (uses saved optimizer state + iteration number)
- Continues training for another 500 iterations
- Saves new games to `./games_session2/` (separate directory)

### Inspect: Visualize what the network sees
```bash
python3 alphazero/main.py \
  --bits 4 \
  --load-games ./games_session1 \
  --inspect ./buffer_inspection_session1.png \
  --inspect-n 8
```

**Output:** `buffer_inspection_session1.png` with 8 random samples showing:
- Colored grid states (which cells have which actions)
- Top-3 actions by probability
- Return value (G_t) for each sample

---

## 📁 File Structure After Training

```
alphazero/
├── games_session1/
│   ├── game_00000_20260422_143022.pt     (~2-3 KB each)
│   ├── game_00001_20260422_143048.pt
│   ├── game_00002_20260422_143115.pt
│   └── ... (500 games total, ≈1.5 MB)
├── games_session2/
│   ├── game_00500_20260422_200000.pt
│   ├── ... (500 more games)
├── checkpoints/
│   ├── checkpoint_00010.pt      (≈4 MB each, saved every 10 iters)
│   ├── checkpoint_00020.pt
│   ├── ...
│   ├── checkpoint_00499.pt
│   └── checkpoint_00999.pt      (final from both sessions)
├── logs/
│   ├── events.out.tfevents.*    (TensorBoard events)
├── *.png                         (inspection images from --inspect)
└── main.py, README.md, etc.
```

---

## 🔍 Inspection Workflow

### After each 50-100 iterations, take a snapshot:

```bash
# After Session 1 is halfway done (iteration 250)
python3 alphazero/main.py --bits 4 \
  --load-games ./games_session1 \
  --inspect ./snapshots/inspection_iter_250.png \
  --inspect-n 8

# After Session 1 completes (iteration 500)
python3 alphazero/main.py --bits 4 \
  --load-games ./games_session1 \
  --inspect ./snapshots/inspection_iter_500.png \
  --inspect-n 8

# After Session 2 is halfway (iteration 750)
python3 alphazero/main.py --bits 4 \
  --load-games ./games_session2 \
  --inspect ./snapshots/inspection_iter_750.png \
  --inspect-n 8
```

**Compare snapshots over time:**
- Early snapshots: mostly random/bad circuits
- Mid snapshots: some structure emerging
- Late snapshots: increasingly sophisticated circuits

---

## 📊 Monitoring During Training

### Terminal 1: Training
```bash
python3 alphazero/main.py --bits 4 --games-dir games --iterations 500
```

### Terminal 2: TensorBoard (live metrics)
```bash
tensorboard --logdir ./logs --port 6006
# Open http://localhost:6006 in browser
```

**Watch these metrics in TensorBoard:**

| Metric | Expected Behavior | Indicates |
|--------|-------------------|-----------|
| `Training/total_loss` | Starts ~4, drops to ~1 | Network learning |
| `Training/policy_loss` | Steady decrease | Policy improving |
| `Training/value_loss` | Steady decrease | Value prediction improving |
| `Training/policy_entropy` | Stays ~3-4 | Exploration maintained |
| `SelfPlay/final_reward` | Starts ~-8, rises to ~-2 | Circuits getting better |
| `Buffer/size` | Rises to 50000 (cap) | Replay buffer full |

---

## 💾 Advanced: Load-Test-Analyze

### Use case: Validate convergence before long training

```bash
# Quick test with Bits=2 (fast)
python3 alphazero/main.py \
  --bits 2 \
  --n-sim 100 \
  --iterations 50 \
  --games-dir test_games

# Inspect results
python3 alphazero/main.py --bits 2 \
  --load-games test_games \
  --inspect test_results.png --inspect-n 4

# If converging well, run with Bits=4
```

---

## ⚡ Optimization Tips

### For Faster Convergence:
```bash
# Increase reliance on network (fewer rollouts)
--alpha-blend 0.8    # Start: 0.5, increase after iteration 20

# More self-play per iteration
--games-per-iter 12  # Default: 5

# More training steps
--train-steps 50     # Default: 20
```

### For Lower GPU Memory:
```bash
# Reduce rollout parallelism
--n-rollouts 32      # Default: 64

# Smaller batch size
--batch-size 128     # Default: 256

# Fewer MCTS simulations
--n-sim 100          # Default: 200
```

---

## 🐛 Troubleshooting

### Q: "Buffer has N samples but B requested"
**A:** Need to wait for buffer to fill. First few iterations won't train. Increase `--games-per-iter`.

### Q: Loss is NaN/Inf
**A:** Learning rate too high. Try `--lr 5e-4` instead of default 1e-3.

### Q: Training is slow
**A:** Use `--alpha-blend 0.8` to reduce rollout calls (18ms → skip).
Or reduce `--n-sim 100` from default 200.

### Q: Want to inspect games but not train
**A:** Use `--inspect` flag:
```bash
python3 main.py --bits 4 --load-games games_v1 --inspect output.png
# Generates PNG and exits (no training)
```

---

## 📈 Expected Training Timeline (Bits=4)

| Phase | Iterations | Wall Time | Loss | Reward | Action |
|-------|-----------|-----------|------|--------|--------|
| Cold-start | 0-50 | 30 min | 4.0 | -8.5 | Run TensorBoard, verify it's working |
| Early learning | 50-150 | 1.5h | 3.0→2.0 | -7→-5 | Check loss is decreasing |
| Main convergence | 150-300 | 3h | 2.0→1.0 | -5→-2 | Take inspection snapshot |
| Fine-tuning | 300-500 | 3h | 1.0→0.5 | -2→-0.5 | Increase --alpha-blend |

**Total for 500 iterations:** ~6-8 hours on GTX 1070 Ti

---

## 🎯 Summary

```bash
# One-liner: multi-session training with inspection
session=1; \
python3 main.py --bits 4 --games-dir games_s${session} \
  $([ $session -gt 1 ] && echo "--load-games games_s$((session-1)) --resume checkpoints/checkpoint_*.pt") \
  --iterations 500 && \
python3 main.py --bits 4 --load-games games_s${session} \
  --inspect snapshot_s${session}.png --inspect-n 8
```

That's it! 🚀
