# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. **Make sure to update this file.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`mario.stimuli/` holds the ROM + state files. On BMRC it was a symlink to `../ppo_study/mario.stimuli` (the repo is intentionally a sibling of `ppo_study` for apples-to-apples comparison with the PPO baseline); on Isambard it's a direct clone of `courtois-neuromod/mario.stimuli`.

`wandb login` is required once before training (wandb is mandatory, not optional).

### Isambard-AI (aarch64/Grace-Hopper) setup notes

This cluster has no Python 3.10 (only system `python3.11`) and no sudo. `stable-retro` has
no wheels on PyPI for any platform (always builds from source) and its **1.0.1 sdist on
PyPI is broken** — missing the whole `src/` directory — so install from GitHub instead:
`pip install "stable-retro @ git+https://github.com/Farama-Foundation/stable-retro.git"`.
That build needs three fixes on this cluster:
- System `gcc` is 7.5.0, too old for `-std=gnu17` used by several libretro cores.
  `module load gcc-native/13.2` (or newer) and export `CC`/`CXX` before building.
- System Python has no dev headers (no `python311-devel`, no sudo to add it). Point CMake's
  `Python_INCLUDE_DIR` at `cray-python/3.11.7`'s headers (`module load cray-python/3.11.7`)
  while still using the venv's Python as the actual interpreter/executable — same ABI, just
  borrowing the headers.
- The vendored LuaJIT is only built with `-fPIC` when `CMAKE_SYSTEM_PROCESSOR` is
  `x86_64`/`AMD64` (see `CMakeLists.txt` around the `BUILD_LUAJIT` block) — aarch64 falls
  through with no PIC flag and the final `_retro...so` link fails. Needs a local patch to
  also pass `-fPIC` for aarch64 before rebuilding LuaJIT.
- Only the `nes` core is needed for this project; the other `add_core(...)` calls
  (snes, genesis, atari2600, gb, gba, pce, 32x, saturn, ds, fbneo, n64) can be commented out
  in `CMakeLists.txt` to cut compile time — they're build-order deps only
  (`add_dependencies(retro-base ${CORE_TARGETS})`), not statically linked.
- **Never compile with high `-j` parallelism on the login node** — it gets reaped
  (`Hangup`/`Terminated` mid-build). Do the build inside a SLURM allocation
  (`salloc -p workq -A <account> --cpus-per-task=32 --mem=64G --time=00:45:00 --no-shell`,
  then `srun --jobid=<id> ...`) instead.
- `torch`/`torchvision` CUDA wheels for aarch64 live on
  `https://download.pytorch.org/whl/cu126/` (plain PyPI `torch` for aarch64 is CPU-only).
- GPUs (NVIDIA GH200) are only on SLURM compute nodes, never the login node — pass
  `--gres=gpu:N` when allocating, or `torch.cuda.is_available()` will always read `False`.
- SLURM submit scripts use `-A brics.u6oz -p workq --gres=gpu:1` and load **no modules at
  runtime** — the venv is self-contained (cu126 torch wheels, imageio-ffmpeg's static
  ffmpeg). Nodes are 4×GH200 / 288 CPUs / 460G, so one GPU's fair share is 72 CPUs + 110G;
  with that allocation `selfplay.num_workers` can go up to ~64 (default is 20).
- The human-data scripts (`fetch_human_data.sh`, `replay_human_bk2.sh`,
  `diag_bk2_download.sh`) additionally need `datalad` + `git-annex` on PATH and the
  courtois-neuromod datasets cloned (paths via `MARIO_SCENES_DIR`/`MARIO_ROOT` env vars).
  On Isambard: datalad is pip-installed in the venv, git-annex (standalone arm64 build)
  lives at `~/tools/git-annex.linux` (add to PATH), and the `mario` dataset with all
  3,374 gamelog .bk2s is cloned at `~/data/mario`. CONP downloads work directly — none
  of BMRC's proxy/CA workarounds apply.

## Human gameplay data (imitation learning)

`scripts/convert_human_bk2.py` converts the CNeuroMod human .bk2 recordings into
replay-buffer-compatible `Trajectory` .npz files: frame-exact emulation through the same
integration/preprocessing/reward shaping as `CustomWrapper`, human button presses
quantised per frame-skip window to the nearest of the 12 discrete actions, multi-life
reps split at deaths into `done_on_life_loss`-style episodes (title-card/respawn frames
skipped via `player_state`: 8=control, 11=dying, 4/5=flagpole). `policies` is the one-hot
human action; `root_values`/`returns` are n-step returns (no MCTS ran); priorities are
uniform. The converted corpus lives in `outputs/human_trajectories/` (~1.9G): 8,766
segments / 3.25M agent-steps / 771 completions across 22 levels (humans never played
w2l2, w7l2, or the castle -4 levels). Subjects present: sub-01/02/03/05/06 (no sub-04),
identity encoded only in filenames. `conversion_report.json` in the same dir has
per-rep provenance.

### Training on human data (`imitation:` config block)

`imitation.enabled=true` turns on two mechanisms (both optional, defaults in
`conf/muzero.yaml`): a supervised **pretrain** phase (`pretrain_steps` gradient steps of
100% human batches — the stored one-hot policies make the policy loss behavioral cloning;
value/reward/consistency losses apply too; reanalyze is forced off learner-side because a
random target net's bootstraps are noise) and a constant **mix** (`mix_ratio` of every RL
batch drawn from a pinned, never-evicted human `TrajectoryBuffer`, wrapped with the
self-play buffer in `MixedBuffer`). `imitation.mix_ratio_schedule` optionally anneals the
mix instead — `[[train_step, ratio], ...]` points, linearly interpolated, thresholds in
RL-phase train steps (the pretrain offset is added automatically in `train_muzero.py`);
e.g. `[[0, 0.25], [70_000, 0.0]]` fades the one-hot BC anchor out instead of dragging on
the policy forever. `train/human_frac` logs the effective ratio. Loader: `src/muzero/human_data.py` — filters by
subject (`imitation.subjects=[sub-01]`, null = all) and level (`levels: match_env`
default filters to `env.levels` via the filename tag, skipping decompression of
filtered-out files). Sequencing in `train_muzero.py`: pretrain runs **before** the
`SelfPlayCoordinator` is constructed, so workers spawn with BC-pretrained weights.
Pretrain steps share the global `training_step` (LR warmup/checkpoints/auto-resume
unchanged); the replay-ratio gate and worker temperature schedule are offset by
`pretrain_steps` — without that offset RL would deadlock at env_step=0. Human samples go
through the same targets/reanalyze/prioritized-replay machinery as self-play data.
`imitation/bc_accuracy` + `imitation/bc_cross_entropy` (held-out human holdout) are the
human-likeness metrics, logged in both phases. **Memory**: obs are (4,96,96) uint8 =
36.9 KB/step, held in RAM — all subjects at the default 12 levels ≈ 1.53M steps ≈ 56.5 GB
(fits one GPU's 110 GB share), one subject ≈ 11 GB; `max_transitions` caps it. Enable
imitation only on fresh `run_name`s — resuming a non-imitation checkpoint below
`pretrain_steps` would BC-pretrain an already-trained net (a warning prints).

## Common Commands

Training (Hydra-based — overrides on CLI):
```bash
# Tiny CPU smoke test (full pipeline in a few minutes on one core)
python scripts/smoke_test.py

# Full local run with overrides
python scripts/train_muzero.py env.levels='[Level1-1]' selfplay.num_workers=1 mcts.num_simulations=10

# SLURM (Isambard-AI, GH200: account brics.u6oz, partition workq)
sbatch scripts/submit_isambard.sh                      # all levels
sbatch scripts/submit_single_level.sh Level1-1         # one level, auto-tags wandb run with slurm id + git sha
sbatch scripts/submit_benchmark.sh                     # inference server benchmark
```

Offline checkpoint replay (no wandb, writes mp4 per level):
```bash
python scripts/replay_checkpoint.py --checkpoint <path>
```

### Eval-time search sweep

`scripts/eval_sweep.py` (+ `scripts/submit_eval_sweep.sh`) sweeps the *eval*
search knobs against a checkpoint and reports completion rates with Wilson 95%
intervals. It exists because greedy `replay/<level>_completed` sat at 0 on runs
whose self-play completion rate was 0.59-0.84, and the two measurements differ
in more than one way at once. `run_replay_rollout` now takes `temperature`,
`root_dirichlet_alpha`/`root_exploration_eps`, `np_seed` and an `info_out` dict
(`final_x`, `timed_out`) so those axes can be varied independently — defaults
reproduce the old fully-greedy replay (no noise, argmax), so per-checkpoint
replay logging is unchanged. Note the two knobs are genuinely independent:
temperature 0.1 is already within a few percent of argmax, so noise (not
temperature) is what separates self-play from greedy eval.

The env has no stochasticity, so a noise-free/argmax cell is one trajectory
regardless of seed — the sweep detects this and runs such cells once, flagging
them `deterministic` in `summary.csv`. See [BACKLOG.md](BACKLOG.md) for the
diagnosis this was built to settle.

### Run lifecycle (chained SLURM jobs)

`submit_chain.sh <run> <N> [overrides]` submits N afterany-dependent legs of
`submit_autocurriculum.sh` that auto-resume from `checkpoints/latest.pt`; the
`SBATCH_EXTRA` env var passes extra sbatch flags to every leg (how the 2-GPU
split is requested). `submit_curriculum.sh [run] [N] [overrides]` wraps it
with the full validated 12-level recipe (2-GPU split, discount 0.999, bonus
200, rescaled LR/temperature schedules, 1M buffer) — see the script header
for the reasoning behind each knob. Three
mechanisms keep the chain sane:
- A final checkpoint is saved when the env-step budget is reached **and** on
  SIGTERM/wall-time (`save_final_checkpoint`), so no training between `save_every`
  boundaries is lost and resumed legs never re-log wandb steps (the old cause of
  "monotonically increasing" warnings and 11-minute re-training legs).
- On reaching `training.total_env_steps` the learner writes
  `outputs/runs/<run>/TRAINING_COMPLETE` (json: env_step/total/training_step);
  `train_muzero.py` fast-exits when the sentinel's env_step already covers the
  configured budget (raise `training.total_env_steps` or delete the sentinel to
  train further), and `submit_autocurriculum.sh` walks + `scancel`s the pending
  dependent legs.
- **Never `put` on a worker-shared mp.Queue from the learner process at shutdown**
  (see `InferenceServer.stop`): terminated workers can die holding the queue's
  write-lock and the learner's QueueFeederThread then deadlocks interpreter exit —
  this zombied SLURM jobs for hours (job kept RUNNING after wandb finished).

Tests:
```bash
pytest                                  # full suite
pytest tests/test_mcts.py               # single file
pytest tests/test_mcts.py::test_name    # single test
```

## Architecture

This is a distributed MuZero implementation. The big-picture flow has three concurrent components communicating via `torch.multiprocessing`:

1. **Learner** (main process, GPU): `src/muzero/muzero.py` — `MuzeroLearner` samples from the replay buffer and runs MuZero loss (value + reward + policy, unrolled `K` steps). Networks live in `src/muzero/networks.py` (`MuZeroNet` = representation + dynamics + prediction, with categorical value/reward supports via `src/muzero/transforms.py`).

2. **Self-play workers** (spawned processes, CPU): `src/selfplay/worker.py` runs gym-retro Mario, performs MCTS (`src/muzero/mcts.py`, `src/muzero/node.py`), and emits trajectories. Workers do **not** own a network — MCTS expansions are RPCs, batched `mcts.leaf_batch` leaves per round trip via virtual visits. The policy training target is the **raw** visit distribution; the temperature schedule only shapes action selection (do not feed the tempered distribution back as the target).

3. **Inference server** (thread inside the coordinator process, GPU): `src/selfplay/inference_server.py` batches MCTS requests from all workers across a single `mp.Queue`, runs them on GPU (optionally AMP), and replies via per-worker `mp.Pipe`s. Learner weight updates are applied to the server's network *in place* — no pickling through pipes. Cadence is `training.weight_broadcast_every`.

The coordinator (`src/selfplay/coordinator.py`) owns the queues/pipes and wires (1)+(2)+(3) together. It also runs replay-evaluation rollouts (`src/selfplay/replay_eval.py`) at each checkpoint to upload mp4s as `wandb.Video`.

Replay buffer (`src/muzero/buffer.py`) is a prioritized trajectory buffer; ingested trajectories keep the worker-computed per-step priorities. Targets (n-step value bootstrap, reward sequence, MCTS policy) are computed in `src/muzero/targets.py` with help from `src/muzero/returns.py`; with `muzero.reanalyze` on, `build_reanalyze_targets` additionally emits bootstrap observations + discount factors so the learner recomputes value targets against a lagged target net at sample time. Action-selection temperature schedule lives in `src/muzero/temperature.py`.

Environment wrappers (`src/env/`) are ported from `ppo_study` — stable-retro NES emulation (`emulation.py`), Mario action set (`mario_actions.py`), and frame preprocessing (`preprocess.py`). The level list is configured via `env.levels` and corresponds to state files in `mario.stimuli/`.

## Config

Hydra config tree rooted at [conf/muzero.yaml](conf/muzero.yaml), with `env: mario` and `model: muzero_mario_small` defaults (`model: muzero_atari` is the full-size paper net — it capped single-GPU self-play at ~5 env-steps/s, which is why the small net is the default). Override anything on the CLI (`section.key=value`). Key knobs:
- `selfplay.num_workers` — must fit CPU budget; each worker is a process.
- `mcts.leaf_batch` — leaves selected (with virtual visits) and evaluated per inference round trip. 1 = fully-sequential paper search; 4 cuts server round trips per env step from 33 to 9 and is the main self-play throughput lever.
- `inference_server.{max_batch,max_wait_ms,use_amp}` — throughput vs. latency tradeoff for the batched GPU server. `max_batch` counts GPU **rows**, not requests (a leaf-batched request is `leaf_batch` rows). `inference_server.compile` (`"off"`/`"default"`/`"reduce-overhead"`) torch.compiles the server forwards — benchmark a short run before enabling on a long chain, and keep `pad_batches=true` with it.
- `inference_server.device` — null = auto: the server's shadow net goes on `cuda:1` when a second GPU is visible (submit with `sbatch --gres=gpu:2 --cpus-per-gpu=72 ...`), separating learner training from self-play inference; with one GPU, behavior is unchanged. Motivation: single-GPU runs managed only ~0.035 gradient steps/env step against the 0.3 cap — the learner loses the GPU to the inference server.
- `muzero.{unroll_K,n_step,discount}` — core MuZero hyperparams.
- `muzero.loss_weight_consistency` — EfficientZero-style SimSiam consistency loss (0 disables): dynamics hidden states are pulled toward stop-grad representations of the real next observations. Big sample-efficiency win since env steps are the bottleneck.
- `muzero.reanalyze` — value reanalyze (EfficientZero-style): n-step value targets are recomputed at sample time by bootstrapping from a lagged target net (synced every `training.target_update_every` train steps) instead of the MCTS root values frozen at collection time. Costs B×(K+1) extra no-grad representation forwards per batch.
- `muzero.{value,reward}_support_*` — support grids live in *transformed* (signed_hyperbolic) space and the model configs interpolate them, so head sizes always match the loss. ±25 value ≈ ±640 raw return, ±8 reward ≈ ±79 raw reward units. Re-check the reward ceiling before raising `env.completion_bonus` past ~500 raw. Changing supports invalidates existing checkpoints (head sizes change).
- `training.max_train_per_env_step` — cap on gradient steps per collected env step (replay-ratio control; also stops the learner from starving the inference server of GPU).
- `training.{lr,lr_min,lr_warmup_steps,lr_decay_steps}` — warmup + cosine-to-floor LR schedule. **Do not reintroduce multiplicative StepLR decay** — it silently drove the LR to 1e-9 by step 600k on a 2-day run and froze learning.
- `training.weight_broadcast_every` — how often the learner's weights are pushed into the inference server.
- `env.done_on_life_loss` — episode ends on first death (true terminal, crisp credit assignment); `env.completion_bonus` — raw reward on stage advance (pre-/10 scaling).
- `worker.torch_threads` / `learner_torch_threads` — kept at 1 to avoid oversubscription across the worker pool.

## Level-completion tracking

The headline goal is finishing levels. Wandb metrics: `selfplay/completed/<level>` (0/1 per episode), `selfplay/completion_rate_100ep` (rolling, all levels pooled), `autocurriculum/completion_rate/<level>`, and `replay/<level>_completed` per replay checkpoint. `checkpoints/best.pt` (+`best.json` sidecar carrying the rate across chain legs) is refreshed on every new rolling-completion-rate high — `step_*.pt` files rotate (keep=10), so best.pt is the only checkpoint guaranteed to survive a late-run performance collapse. Note `replay/<level>_completed` evaluates *greedy* MCTS (temp 0, no Dirichlet) — it can sit at 0 while noisy self-play completes 30-70%; the gap means the policy still needs exploration noise to get past specific stuck-spots. The autocurriculum samples levels ∝ inverse-mean-episode-length × incompletion-rate, so mastered levels free up worker time for unfinished ones (`autocurriculum.min_weight` floor guards against forgetting).

## Logging

wandb is required (`wandb login` once). Per-step losses, per-episode self-play returns per level, and per-checkpoint mp4 rollouts (`wandb.Video`) are all logged. `submit_single_level.sh` auto-tags every run with SLURM job id, level, git branch, and git sha so wandb runs are traceable back to the code version.

**Search-quality diagnostics.** `selfplay/mcts_prior_entropy/<level>`,
`selfplay/mcts_visit_entropy/<level>` and `selfplay/mcts_visit_max_frac/<level>`
are per-episode means of the root policy-head entropy, the raw visit-distribution
entropy and the largest single-action visit share. Entropies are nats — read them
against ln(12) = 2.485 for the 12-action Mario set; a prior entropy sitting near
that bound means the policy head is near-uniform and root Dirichlet noise is doing
the exploring. Prior entropy is deliberately sampled **before** noise is added, so
it measures the head itself. Plumbing is `MCTS.run(..., stats_out=dict)`, opt-in:
omitting it leaves the return signature and the eval/benchmark call sites unchanged.

Note: bare `pytest` collects `.venv/` and fails with ~113 collection errors —
always scope it (`pytest tests/`).

## Reference

Structural reference (pattern source, not a dependency): `/well/costa/users/zqa082/Muzero-Hanoi`.
