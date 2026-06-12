# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. **Make sure to update this file.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`mario.stimuli/` is a symlink to `../ppo_study/mario.stimuli` (ROM + state files). The repo is intentionally designed as a sibling to `ppo_study` for apples-to-apples comparison with the PPO baseline.

`wandb login` is required once before training (wandb is mandatory, not optional).

## Common Commands

Training (Hydra-based — overrides on CLI):
```bash
# Tiny CPU smoke test (full pipeline in a few minutes on one core)
python scripts/smoke_test.py

# Full local run with overrides
python scripts/train_muzero.py env.levels='[Level1-1]' selfplay.num_workers=1 mcts.num_simulations=10

# SLURM (BMRC, A100 80GB)
sbatch scripts/submit_bmrc.sh                          # all levels
sbatch scripts/submit_single_level.sh Level1-1         # one level, auto-tags wandb run with slurm id + git sha
sbatch scripts/submit_benchmark.sh                     # inference server benchmark
```

Offline checkpoint replay (no wandb, writes mp4 per level):
```bash
python scripts/replay_checkpoint.py --checkpoint <path>
```

Tests:
```bash
pytest                                  # full suite
pytest tests/test_mcts.py               # single file
pytest tests/test_mcts.py::test_name    # single test
```

## Architecture

This is a distributed MuZero implementation. The big-picture flow has three concurrent components communicating via `torch.multiprocessing`:

1. **Learner** (main process, GPU): `src/muzero/muzero.py` — `MuzeroLearner` samples from the replay buffer and runs MuZero loss (value + reward + policy, unrolled `K` steps). Networks live in `src/muzero/networks.py` (`MuZeroNet` = representation + dynamics + prediction, with categorical value/reward supports via `src/muzero/transforms.py`).

2. **Self-play workers** (spawned processes, CPU): `src/selfplay/worker.py` runs gym-retro Mario, performs MCTS (`src/muzero/mcts.py`, `src/muzero/node.py`), and emits trajectories. Workers do **not** own a network — every MCTS expansion is an RPC.

3. **Inference server** (thread inside the coordinator process, GPU): `src/selfplay/inference_server.py` batches MCTS requests from all workers across a single `mp.Queue`, runs them on GPU (optionally AMP), and replies via per-worker `mp.Pipe`s. Learner weight updates are applied to the server's network *in place* — no pickling through pipes. Cadence is `training.weight_broadcast_every`.

The coordinator (`src/selfplay/coordinator.py`) owns the queues/pipes and wires (1)+(2)+(3) together. It also runs replay-evaluation rollouts (`src/selfplay/replay_eval.py`) at each checkpoint to upload mp4s as `wandb.Video`.

Replay buffer (`src/muzero/buffer.py`) is a prioritized trajectory buffer; targets (n-step value bootstrap, reward sequence, MCTS policy) are computed in `src/muzero/targets.py` with help from `src/muzero/returns.py`. Action-selection temperature schedule lives in `src/muzero/temperature.py`.

Environment wrappers (`src/env/`) are ported from `ppo_study` — stable-retro NES emulation (`emulation.py`), Mario action set (`mario_actions.py`), and frame preprocessing (`preprocess.py`). The level list is configured via `env.levels` and corresponds to state files in `mario.stimuli/`.

## Config

Hydra config tree rooted at [conf/muzero.yaml](conf/muzero.yaml), with `env: mario` and `model: muzero_mario_small` defaults (`model: muzero_atari` is the full-size paper net — it capped single-GPU self-play at ~5 env-steps/s, which is why the small net is the default). Override anything on the CLI (`section.key=value`). Key knobs:
- `selfplay.num_workers` — must fit CPU budget; each worker is a process.
- `inference_server.{max_batch,max_wait_ms,use_amp}` — throughput vs. latency tradeoff for the batched GPU server.
- `muzero.{unroll_K,n_step,discount}` — core MuZero hyperparams.
- `muzero.loss_weight_consistency` — EfficientZero-style SimSiam consistency loss (0 disables): dynamics hidden states are pulled toward stop-grad representations of the real next observations. Big sample-efficiency win since env steps are the bottleneck.
- `training.max_train_per_env_step` — cap on gradient steps per collected env step (replay-ratio control; also stops the learner from starving the inference server of GPU).
- `training.{lr,lr_min,lr_warmup_steps,lr_decay_steps}` — warmup + cosine-to-floor LR schedule. **Do not reintroduce multiplicative StepLR decay** — it silently drove the LR to 1e-9 by step 600k on a 2-day run and froze learning.
- `training.weight_broadcast_every` — how often the learner's weights are pushed into the inference server.
- `env.done_on_life_loss` — episode ends on first death (true terminal, crisp credit assignment); `env.completion_bonus` — raw reward on stage advance (pre-/10 scaling).
- `worker.torch_threads` / `learner_torch_threads` — kept at 1 to avoid oversubscription across the worker pool.

## Level-completion tracking

The headline goal is finishing levels. Wandb metrics: `selfplay/completed/<level>` (0/1 per episode), `selfplay/completion_rate_100ep` (rolling, all levels pooled), `autocurriculum/completion_rate/<level>`, and `replay/<level>_completed` per replay checkpoint. The autocurriculum samples levels ∝ inverse-mean-episode-length × incompletion-rate, so mastered levels free up worker time for unfinished ones (`autocurriculum.min_weight` floor guards against forgetting).

## Logging

wandb is required (`wandb login` once). Per-step losses, per-episode self-play returns per level, and per-checkpoint mp4 rollouts (`wandb.Video`) are all logged. `submit_single_level.sh` auto-tags every run with SLURM job id, level, git branch, and git sha so wandb runs are traceable back to the code version.

## Reference

Structural reference (pattern source, not a dependency): `/well/costa/users/zqa082/Muzero-Hanoi`.
