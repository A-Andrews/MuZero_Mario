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
- **Storage: home has a hard 101 GB quota** (`quota -s`; no soft limit, no grace) and
  hitting it killed the entire first T6 fleet on 2026-08-12 — 12 jobs each writing a
  271 MB checkpoint every 16 min died simultaneously at their next save boundary with
  exit 1 and *empty* `.err` files (wandb's console redirect swallowed the tracebacks;
  `WANDB_CONSOLE=off` is now exported by the submit scripts so this can't recur).
  `outputs/` is therefore a **symlink to `/projects/u6oz/atdandrews/MuZero_Mario/outputs`**
  (200 TB Lustre, no quota set) — all run dirs/checkpoints land there via the existing
  relative paths. Put any new large artifacts under `/projects/u6oz/atdandrews/` too,
  never on home. `training.checkpoint_keep` (default 10) controls step_*.pt rotation,
  and a failed checkpoint save now warns and continues instead of killing the run.
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
for the reasoning behind each knob. `submit_specialist.sh <LEVEL> [N]` is the
single-level (T6) sibling, and `submit_specialist_imitation.sh <LEVEL> [N]` is
the human-demo rescue variant for levels self-play never completes — same
recipe plus `imitation.enabled` and the completion-gated eps anneal, on a
fresh `spec-imit-<level>` run name (imitation must only ever be enabled on a
fresh run_name). Three mechanisms keep the chain sane:
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
- **`submit_chain.sh` warns when N_JOBS cannot reach the env-step budget.** It
  reads the leg script's own `#SBATCH --time`, takes `training.total_env_steps`
  from the overrides (else the config), assumes ~110 env-steps/s
  (`MUZERO_ENV_STEPS_PER_SEC` overrides) and prints the leg count actually
  needed, subtracting progress already banked in `latest.pt` so a resumed chain
  is not told to re-budget the whole run. `MUZERO_CHAIN_DRY_RUN=1` runs the
  check and submits nothing. Why: T7 was submitted with the default 4 legs
  against 60M env steps and stopped 4 days later at 39.4M/36.0M. Every leg
  exited `TIMEOUT 0:0` — the designed resume path — so nothing looked wrong:
  no sentinel, no error, no queue entry, `.err` files holding only wandb
  banners. The sentinel mechanism cancels *surplus* legs; nothing detected the
  shortfall. **A chain out of legs is indistinguishable from a finished run at
  a glance — check `TRAINING_COMPLETE` exists, not just that the queue is
  empty.**
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

Replay buffer (`src/muzero/buffer.py`) is a prioritized trajectory buffer; ingested trajectories keep the worker-computed per-step priorities. Targets (n-step value bootstrap, reward sequence, MCTS policy) are computed in `src/muzero/targets.py` with help from `src/muzero/returns.py`; with `muzero.reanalyze` on, `build_reanalyze_targets` additionally emits bootstrap observations + discount factors so the learner recomputes value targets against a lagged target net at sample time. Action-selection temperature schedule lives in `src/muzero/temperature.py` (piecewise **constant** — steps down at thresholds). `src/muzero/schedules.py` is the piecewise-**linear** counterpart shared by `imitation.mix_ratio_schedule` and `mcts.root_exploration_eps_schedule`.

Environment wrappers (`src/env/`) are ported from `ppo_study` — stable-retro NES emulation (`emulation.py`), Mario action set (`mario_actions.py`), and frame preprocessing (`preprocess.py`). The level list is configured via `env.levels` and corresponds to state files in `mario.stimuli/`.

**`conf/env/mario.yaml`'s 12 levels (worlds 1-4, stages 1-3) are a default, not the scope.** That list predates the git history — it is absent from the initial commit and appears already formed in f91adf7, most likely inherited from the `ppo_study` PPO baseline. It was therefore chosen to serve a PPO comparison, not the brain-data one, and it covers only half the 22 levels with human recordings. T9 (2026-09) trained specialists for the missing worlds 5-8; `mario.stimuli/SuperMarioBros-Nes/` ships state files for all 34 levels including the castle -4s and the 5-x/6-x warps.

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
- `mcts.root_exploration_eps_schedule` — **root-Dirichlet anneal** (T2.1).
  `[[train_step, eps], ...]` knots, linearly interpolated, clamped outside the
  range; `null` (the default) keeps `mcts.root_exploration_eps` constant, which
  is what every run before 2026-08-11 did. Thresholds are RL-phase train steps —
  the coordinator already subtracts the imitation pretrain offset from the
  workers' counter, so no manual offset is needed. Size it per run alongside
  `selfplay.temperature_schedule` (`submit_specialist.sh` uses
  `[[0,0.25],[500000,0.05]]`, bottoming out where temperature does). Why it
  matters: with eps pinned at 0.25 the policy head is never forced to stand on
  its own, so `selfplay/completion_rate_100ep` overstates deployed performance —
  survivable for a specialist you always evaluate with noise on, fatal for T8,
  where those checkpoints become distillation teachers. The effective value is
  logged as `selfplay/root_exploration_eps`; read completion rate against it.
- `mcts.root_exploration_eps_gate_on_completion` — **gates that anneal on the
  run's first completion** (default false = the T6 behaviour). When true the
  schedule's step axis becomes `train_step - first_completion_step`: eps holds
  at the schedule's first knot until self-play completes a level even once,
  then anneals at the configured rate. The origin is persisted to
  `checkpoints/first_completion.json` (a sidecar, not the checkpoint, so
  enabling the gate doesn't invalidate existing checkpoints) and reloaded on
  resume, so a chained leg never re-arms the gate; delete that file to re-arm.
  The sidecar is written whether or not the gate is on, as provenance for when
  a run escaped. Why it exists: the T6 Level1-3/Level4-3 specialists ran ~97k
  and ~107k episodes with **zero** completions, both pinned at a pit gap
  (Level1-3: 73% of episodes end x=750-999 by death, not timeout), with
  `mcts_root_q_mean` flat all run (27.4 → 27.3) where Level3-2 tripled it
  (39.8 → 125.3). A pit gap is a discontinuous reward cliff, so "walk to the
  edge and stop" is a real local optimum the value head correctly certifies;
  escape needs one lucky deep excursion, and annealing exploration away from a
  policy that has never seen the reward closes that door permanently.
- `env.noop_max` / `env.skip_to_control` — **stochastic starts**. NES Mario is
  otherwise fully deterministic, which made every greedy rollout a single
  trajectory rather than a sample (the T1 sweep's eval cells were all n=1 and
  flipped outcome on single knobs) and never pressured the policy to be robust.
  Each reset burns a uniform random `0..noop_max` NOOP frames. **The two knobs
  are a package**: every level opens with a scripted intro that ignores input
  (measured 123 frames on Level1-1, 117 on Level1-2; `player_state` goes
  0 → 7 → 8 = "in control"), so with `skip_to_control=false` the whole delay is
  absorbed and `final_x` is bit-identical across seeds — verified. With it on,
  trajectories genuinely diverge. Side effect: the intro frames leave the
  trajectory, so every episode loses ~30 agent steps of uncontrollable title
  card, which also shifts the autocurriculum's inverse-length weighting.
  `run_replay_rollout` defaults both **off** regardless of the training config,
  so greedy eval stays a single reproducible trajectory and stays comparable to
  the deterministic-env diag baselines; pass them explicitly to evaluate under
  the training start distribution.
- `worker.torch_threads` / `learner_torch_threads` — kept at 1 to avoid oversubscription across the worker pool.

### Lesion study (`src/muzero/lesion.py`, `scripts/lesion_eval.py`)

Re-initialise a targeted component of a trained net at evaluation time and
measure the performance deficit — the Mario counterpart of the Towers-of-Hanoi
region-specific-planning experiment at
`~/region-specific-planning/Muzero-Hanoi` (value → PFC, policy → cerebellar,
compared against human lesion data). Three conventions are shared with that
study and must not drift, or the two are no longer comparable:

1. **A lesion is random re-initialisation**, not added noise and not zeroing.
   Here each leaf module's own `reset_parameters()` is called, so a lesioned
   Conv2d/Linear/BatchNorm2d gets exactly its constructor-time distribution.
2. **Evaluation time only.** Never retrain after lesioning for a main result.
3. **Lesioning one target leaves every other parameter bit-identical** —
   `tests/test_lesion.py::TestLesionContract`, 11 tests.

**The Mario-specific trap.** Hanoi's policy/value/reward heads are standalone
`nn.Linear` stacks off the latent, so each is trivially separable. Mario's
`PredictionNet` puts policy and value behind a **shared residual trunk**
(`prediction.blocks`) and `DynamicsNet` carries the reward head. A "policy
lesion" written as a `prediction`-wide reset therefore also destroys value and
reports a policy deficit that is really policy+value. `LESION_TARGETS` resets
only the head-specific conv/bn/fc; the trunk is separately lesionable as
`pred_trunk`. Two targets have no Hanoi analogue: **`transition`** (`dynamics`
minus its reward head — the latent forward model MCTS actually rolls out) and
**`encoder`**. `projection_net` is deliberately not a target: it is a
learner-only SimSiam head no inference path touches, so lesioning it is a
guaranteed no-op that would read as a spurious "no deficit".

`apply_lesion(net, targets, seed)` saves/seeds/restores the torch RNG, so a
lesion is reproducible without shifting the stream rollout seeds come from.
**A lesion is a random variable** — one re-init is one sample — so
`--lesion-seeds` repeats each condition with independent re-inits and results
should be read across seeds.

`scripts/lesion_eval.py` (via `sbatch scripts/submit_lesion_eval.sh`) scores
conditions by greedy run-throughs, reporting `final_x` alongside completion:
completion is 0/1 and most lesions drive it to zero, whereas distance degrades
gradually and separates a mild deficit from a total one. Every rollout rebuilds
the net from the checkpoint, so lesions never stack. `--sims` sweeps
`mcts.num_simulations`, which is how you ask whether a component is only
load-bearing when the search is deep enough to consult it.

**Findings so far (2026-09-07, Level3-3, jobs 6380549 / 6380994)** — see
BACKLOG.md T10 for the tables:
- **The deficit is close to inverse to the damage.** Resetting the policy
  head's **1,266** parameters (0.006% of 22.6M) takes the model from finishing
  to dying at x=357; resetting the **7.0M**-parameter forward model still
  reaches x=1063.
- **Deeper search amplifies a broken value head.** Value-lesioned completion
  falls 1.00 → 0.50 as simulations go 10 → 200, while intact sits at ceiling
  throughout. More planning is actively worse than less when the bootstrap is
  corrupted.
- **`reward` is free at every search depth** — the one component this agent
  does not appear to use.

### Shipping models to collaborators (`package_models.py`)

`sbatch scripts/submit_package_models.sh [args]` builds a self-contained zip a
collaborator can unzip and use with **nothing from this repo and no
stable-retro/ROM install** — written for the CNeuroMod brain-data comparison.
Output goes to `/projects/u6oz/atdandrews/MuZero_Mario/exports/`, never home.

What it does: strips optimizer/scheduler/RNG state from each level's chosen
checkpoint (271 MB → 90 MB, 49 MB zipped), copies `networks.py`,
`transforms.py` and `preprocess.py` under a `code/src/...` tree (with an
**empty** `src/env/__init__.py`, since the real one imports the whole emulation
stack), and emits `manifest.json` (per-level run, train/env step, self-play
rate, greedy run-through stats, sha256), a standalone `load_model.py` and a
README. Templates for the last two are `scripts/_bundle_{load_model.py,README.md}`.

- **Level selection is derived, not hard-coded.** Levels come from the run dirs,
  and per level the `spec-<level>` and `spec-imit-<level>` runs compete on
  recorded `best.json` rate. `eval_human_benchmark.py` and both
  `submit_specialist*.sh` scripts do the same (the latter derive valid levels
  from the shipped `.state` files), so a new fleet appears everywhere without
  editing five lists. **This was a real bug source** — three separate
  hard-coded 12-level lists silently excluded worlds 5-8.
- `--include-incomplete` ships a level whose runs never completed, falling back
  to `latest.pt` and marking the entry `completed_level: false` so the manifest
  never passes it off as a best. `--pin LEVEL=RUN` forces a run.
- **Two traps worth knowing when anyone loads these checkpoints**: the runs are
  192-channel/10-dynamics-block nets, *not* the library defaults, so always
  build from the checkpoint's own `cfg_snapshot["model"]`; and obs are stored
  uint8 but the model wants `obs.float()/255.0`. `load_model.py` does both.
- **The confound that matters for brain comparison**: `spec-imit-*` models were
  BC-pretrained on the same CNeuroMod subjects' gameplay the fMRI comes from,
  so they cannot serve as independent evidence that a model's representations
  resemble those subjects' brains. The bundle README flags them as a separate
  group; say it explicitly when sending.

**Known gap, unfixed — but it does not block the collaboration.** Confirmed
2026-09-07 that the collaborator feeds their own stimulus frames, so they need
`load_model.frames_to_obs()` (which reproduces this pipeline exactly) and not
our corpus. The gap below is therefore an *internal* limitation, relevant only
if we want TR-aligned analysis from the corpus ourselves:
`outputs/human_trajectories/` cannot be aligned frame-accurately to the fMRI. Segments split at deaths, title-card/respawn
frames are dropped, and no absolute .bk2 frame index is stored — only
per-segment step counts in `conversion_report.json` — so cumulative agent steps
do not map linearly onto the .bk2 timeline. Fixing it means emitting a
frame-index array from `convert_human_bk2.py` and re-running the conversion.

### The human run-through benchmark (`images/` figure job)

`sbatch scripts/submit_human_benchmark.sh` plays each level's **best** checkpoint
through the level a few times and plots those run-throughs against the human
ones, writing `images/human_vs_agent_runthrough.{pdf,json}` into the repo.
`scripts/eval_human_benchmark.py` is the driver and runs standalone
(`--device cpu --levels Level1-1 --rollouts 1` for a quick check).

It answers a deliberately narrower question than the in-run `human_compare:`
metrics: *can the model get through the level at all, in a human-like number of
steps* — not a like-for-like rate comparison. Consequences of that framing:
- **`best.pt`, not `latest.pt`.** `step_*.pt` rotates (keep=10) and several runs
  end in a late collapse, so the final checkpoint understates what was learned.
  Every figure row and JSON record carries the run name, checkpoint path and
  training step, so which model produced a point is never ambiguous.
- **The run is chosen by score, not by family.** Levels with both a T6
  `spec-<level>` and a T7 `spec-imit-<level>` take whichever has the higher
  recorded `best.json` rate, so the imitation rescues win 1-3 and 4-3 on merit
  rather than by a hard-coded rule.
- **Stochastic starts are ON** (read from the run's own `env.noop_max` /
  `env.skip_to_control`), because the env is otherwise deterministic and N greedy
  rollouts would be N copies of one trajectory. `--deterministic` gives the
  single reproducible greedy trajectory instead, and correctly collapses to one
  rollout.
- The human side is the p10-p90 spread of **successful** human run-throughs with
  a median tick; levels humans never played (2-2 in the training set) get no band
  and are labelled as such.

`images/` is git-tracked (see its README) — the PDF is vector and tens of KB, so
committing it versions the figure alongside the code that made it. matplotlib is
a dependency of this script only; the submit script points `MPLCONFIGDIR` at
`outputs/` so no font cache is written to the quota-limited home.

### Comparing each new model against the human (`human_compare:` config block)

Every checkpoint replay is scored against a human playing the **same level**,
using the converted corpus in `outputs/human_trajectories/`. On by default and
**independent of `imitation.enabled`** — a pure self-play run gets the
comparison too. Implementation: `src/muzero/human_baseline.py`, called from
`MuzeroLearner._run_replay`, so it rides the replay cadence
(`training.replay_every_train_steps`, default 50k — *not* every 5k checkpoint)
because the performance half needs an actual agent rollout to compare.

Metrics logged alongside the existing `replay/<level>_*` keys:
- `human/<level>_{completion_rate,no_death_rate,length_median,length_p10,return_median}`
  — flat reference lines. `completion_rate` is **per rep** (one human attempt,
  comparable to a self-play episode); `no_death_rate` is the stricter
  finished-on-the-first-life rate. Lengths/returns come from the completing
  segment; `length_p10` is a "good human" rather than a median one.
- `compare/<level>_{completed_vs_human,return_ratio,length_ratio,beats_human_median,beats_human_p10}`
  — the performance comparison. `length_ratio` is only emitted when the agent
  actually reached the flag (time-to-flag is meaningless otherwise); **lower is
  better** there, unlike every other ratio.
- `compare/<level>_{action_agreement,action_cross_entropy}` — the behavioural
  half: the checkpoint's policy head scored against the human's *actual* button
  press on held-out human states. This is the per-level, per-checkpoint sibling
  of the corpus-wide `imitation/bc_accuracy`.

Three things that are easy to get wrong here:
- **Reward-scale rebasing.** The converter baked in its own
  `COMPLETION_BONUS = 100` while the curriculum recipe runs
  `env.completion_bonus=200`, so raw returns are not comparable as stored.
  `HumanLevelStats.return_at_bonus()` re-bases them — sound because the bonus is
  a single additive term on the completing step and the replay metric is an
  *undiscounted* sum (so the converter's `DISCOUNT = 0.997` never enters).
  If you change the converter's constants, update `CONVERTER_COMPLETION_BONUS`.
- **Leakage.** When `imitation.enabled`, action-agreement is scored only on the
  files `load_human_buffer` held out. The holdout is a seeded shuffle of the
  *filtered* file list, so `split_human_files` (shared by both callers) must be
  given the imitation loader's **exact** level/subject filters — filtering to
  one level after the split is fine, re-splitting per level is not. That is why
  `human_compare.subjects` is ignored (with a warning) when it disagrees with
  `imitation.subjects`.
- **The two sides are not measured the same way, and that is the point.** The
  agent number is one greedy, deterministic-start rollout; the human numbers are
  distributions over many attempts. Read `completed_vs_human` as "did this model
  finish it" against "how often did a person", not as a like-for-like delta.

Per-level stats are memoised to `outputs/human_trajectories/human_level_stats.json`
(keyed by level + subject set; bump `_CACHE_VERSION` if the schema changes) — a
full scan is ~0.2 s/level and reads only the small npz members, never `obs_stacks`.
The action-eval obs *are* held in RAM: 36.9 KB/step, so the default
`action_eval_transitions_per_level: 1024` is ~38 MB/level (~450 MB across 12
levels); set it to 0 to keep only the performance comparison. Both are loaded
lazily inside the background replay thread, so learner startup is untouched.
Levels humans never played (w2l2, w7l2, the castle -4s) are simply skipped.

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

`pytest tests/` used to die silently part-way through on the login node (it
looked like a hang around the inference-server tests). The cause was OpenMP
thread exhaustion, not the tests: `export OMP_NUM_THREADS=1` (plus
`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`) runs the whole 125-test suite there in
~9s. The same export is what lets `scripts/smoke_test.py`'s train job run on the
login node — without it the learner dies at
`libgomp: Thread creation failed: Resource temporarily unavailable`.

**Replay videos cannot be written on the login node**, and `OMP_NUM_THREADS=1`
does not help — ffmpeg spawns its own encoder threads and hits the same
`ulimit -u` wall (1900 on Isambard, against ~6000 threads already live for the
user). It surfaces as `pthread_create() failed: Resource temporarily
unavailable` → `Could not open encoder before EOF` → a 0-byte mp4 and
`[replay] <level> video failed`. This is login-node-only: all 754 mp4s from
real compute-node runs are valid. Since only the *video* dies, `_run_replay`
logs the replay scalars (and the human comparison) before touching the encoder
and falls back to a plain `wandb.log` on encoder failure, flagging
`replay/<level>_video_error` — a smoke run on the login node therefore still
exercises and logs the full metric path.

## Reference

Structural reference (pattern source, not a dependency): `/well/costa/users/zqa082/Muzero-Hanoi`.
