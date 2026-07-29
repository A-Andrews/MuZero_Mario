# MuZero-Mario backlog

**Status: T1 running as of 2026-07-29.** The two eval sweeps are queued on SLURM
(jobs **5825183** Level1-2, **5825184** Level1-1, 3h wall each). VGDL still holds
priority for everything else. Both diagnostic runs finished cleanly; no work is
at risk.

Next action: **read the T1 results** when the jobs land, then pick T2 or T3 by
the decision table under T1.

---

## Where things stand

| Run | Steps | Peak self-play completion | Greedy replay | Status |
|---|---|---|---|---|
| `level1-1-diag-v1` | 15M / 780K train | **0.84** @ step 696K | **0** at every checkpoint | done 2026-07-18 |
| `level1-2-diag-v1` | 15M / 763K train | **0.68** @ step 698K | **0** at every checkpoint | done 2026-07-21 |
| T1 eval sweep (both levels) | — | — | — | **submitted 2026-07-29**, jobs 5825183/5825184 |
| `curriculum-v1` (12 levels, 60M) | — | — | — | **never submitted**, blocked on T2 |

The discount fix (0.997 → 0.999 + `completion_bonus` 200) is confirmed and is
what unlocked completions at all; every pre-fix run was ≤4%.

### The open problem

Greedy replay eval completes 0 on both levels while noisy self-play completes
59-84%. The cause is narrowed down:

- By the end of both runs the self-play temperature schedule was already at
  **0.1** (`[[0,1],[100000,0.5],[300000,0.25],[500000,0.1]]`), i.e. visits^10 —
  within a few percent of argmax.
- Greedy eval is temp 0 argmax **and** `root_exploration_eps=0`
  (was hardcoded in `replay_eval.py`).
- So the whole 0.59-vs-0.00 gap is attributable to **root Dirichlet noise**
  (`eps=0.25, alpha=0.25`), not to temperature.
- Supporting: per-step policy CE ≈ 1.97 nats vs ln(12)=2.48 uniform (the losses
  logged are `init + (1/K)·recur`, so ~2× per-step) — the prior is barely more
  informative than uniform. Per-step value CE ≈ 2.37 vs ln(201)=5.30.
- Supporting: the sims sweep (50→200→400, job 5727108) gave **bit-identical**
  returns on Level1-1 across three seeds. More search over an undiscriminating
  model converges harder on the same wrong action.

Read: root noise is not just exploring, it is carrying the performance. The
0.59 rate is partly randomized search through a chokepoint rather than a policy
that knows the answer.

Level1-2's greedy failure is concrete: dies at x=850 (~25% in) into a cluster of
three Koopas, at frame ~250/283 of
`outputs/runs/level1-2-diag-v1/videos/step_750000/Level1-2.mp4`.

---

## T1 — Run the eval sweep (**submitted 2026-07-29 — awaiting results**)

Code committed in `684e829` (82/82 tests pass, smoke-tested). ~1 GPU-hour each.
Decides which of T2/T3 is worth doing, and whether the curriculum run should
launch as-configured.

Submitted as:

```bash
sbatch scripts/submit_eval_sweep.sh \
    outputs/runs/level1-2-diag-v1/checkpoints/best.pt Level1-2   # job 5825183
# and, for the level that times out rather than dies:
sbatch scripts/submit_eval_sweep.sh \
    outputs/runs/level1-1-diag-v1/checkpoints/best.pt Level1-1   # job 5825184
```

Results will land in `outputs/eval_sweep/Level1-2_level1-2-diag-v1_5825183/`
and `outputs/eval_sweep/Level1-1_level1-1-diag-v1_5825184/`; logs in
`logs/muzero_eval_sweep-{5825183,5825184}.{out,err}`.

**Results: _pending — fill in when the jobs land._**

Grid: eps {0, 0.1, 0.25} × temperature {0, 0.1, 0.25} × `pb_c_init` {1.25, 2.5},
20 episodes/cell with Wilson 95% intervals. Noise-free + argmax cells are
detected as deterministic and run once (the env has no stochasticity, so
repeats would be bit-identical). Writes `results.json` + `summary.csv` +
one mp4 per cell to `outputs/eval_sweep/<level>_<run>_<jobid>/`.

**What the outcomes mean:**
- *eps>0 cells recover ~0.6 completion, eps=0 cells stay at 0* → noise is a
  crutch; the policy/value are not discriminative. Do **T2**.
- *`pb_c_init=2.5` recovers it without noise* → it was a search-exploration
  shortfall, not a model-quality problem. Cheap fix: raise `pb_c_init`.
- *All cells near 0* → the 0.59 self-play number is measuring something other
  than deployable skill (e.g. completions concentrated in a minority of workers
  or lucky episodes). Re-examine how `completion_rate_100ep` is pooled before
  anything else.

## T2 — Training changes (blocked on T1)

Ranked. All are config/schedule-level except the third.

1. **Anneal root Dirichlet eps** 0.25 → ~0.05 alongside the temperature
   schedule. Currently eps never decays, so the policy head is never forced to
   stand on its own and `completion_rate_100ep` is not an honest estimate of
   deployed performance. Needs a schedule mechanism like
   `src/muzero/temperature.py`, threaded into `worker.py:107`.
2. **Raise self-play `mcts.num_simulations`** 50 → 100+. Sharper visit
   distributions are the direct fix for a policy head sitting near uniform —
   the training *target* is currently too diffuse. Costs throughput; the
   untested 2-GPU split is the headroom.
3. **Stochastic starts** — random 0-30 no-op frames at reset. The env is fully
   deterministic today, so the policy is never pressured to be robust and
   greedy eval has zero variance (hence the bit-identical seeds). Standard
   practice; makes every eval number meaningful. Env-wrapper change in
   `src/env/env.py:reset`.

## T3 — Capacity / horizon (only if T2 doesn't close the gap)

- `muzero.unroll_K` 5 → 10. At `frame_skip=4` the learned model currently sees
  20 frames of lookahead — thin for the frame-precise enemy avoidance that the
  x=850 Koopa cluster needs.
- Larger net. Note both diag runs already used `muzero_mario_medium`
  (192ch/10 dyn); the only bigger option in the tree is `muzero_atari`
  (256ch/15), flagged in CLAUDE.md as capping self-play at ~5 env-steps/s.
- Imitation anchored on Level1-2. **Rated low**: the human corpus has 347 w1l2
  segments but only **11 completed** ones (vs 63 for w1l1) — thin supervision
  for the exact behavior wanted, though the 153K human steps still help
  representation learning.

## T4 — The curriculum run (blocked on T1/T2)

`scripts/submit_curriculum.sh` was written 2026-07-20 and committed in `ed4808d`
but **never submitted**. 12 levels, 60M env steps, 4 chained 24h legs, 2 GPUs,
64 workers, 1M buffer.

**Do not launch as-is.** It bakes in `eps=0.25` forever and 50 sims — exactly
the configuration T1 is testing. Rerun the header's recipe after T2 lands.

When it does go out, run the benchmark leg first — the 2-GPU split and the
64-worker throughput have never run in production:

```bash
bash scripts/submit_curriculum.sh bench-2gpu 1 training.total_env_steps=200_000
```

## T5 — Housekeeping

- [x] Commit the T1 work: `scripts/eval_sweep.py`,
      `scripts/submit_eval_sweep.sh`, the `replay_eval.py` knobs, this file,
      and the CLAUDE.md section. Done 2026-07-29 in `684e829` on `isambard`.
- [x] `mario.stimuli` shows as modified in `git status` (submodule/clone drift) —
      benign, leave it. The Isambard datalad clone replaced git-annex symlinks
      with plain `/annex/objects/MD5E-...` pointer files, so every `.state` reads
      as content-modified. No real content change; nothing to commit.
- [x] Consider logging policy-head entropy and MCTS visit concentration during
      training. Both diagnoses above had to be reconstructed from loss
      magnitudes after the fact; these two scalars would have shown it live.
      Done 2026-07-29: `MCTS.run(..., stats_out=dict)` (opt-in, so the eval and
      benchmark call sites are untouched) reports `prior_entropy`,
      `visit_entropy`, `visit_max_frac` and the `uniform_entropy` bound; the
      worker means them per episode and the learner logs
      `selfplay/mcts_{prior_entropy,visit_entropy,visit_max_frac}/<level>`.
      Prior entropy is taken **before** Dirichlet noise on purpose — the
      question is whether the policy head is discriminative on its own, and
      noise would push it straight back to uniform. Read against ln(12)=2.485.

---

## Incidental findings worth keeping

- **Level1-2 has a ~29-agent-step scripted intro** (Mario descending the
  entry pipe, ~116 frames) during which `player_x_pos` reads 0 and no reward
  accrues — ~10% of a typical 300-step episode is uncontrollable. Not a bug,
  but relevant when reading episode lengths and when setting `max_steps`.
- The login node kills multi-threaded torch (`libgomp: Thread creation failed`).
  Anything beyond a trivial single-threaded script needs `salloc`/`sbatch`;
  `OMP_NUM_THREADS=1` is enough for quick import-level checks.

---

## Activity log

Newest first. One line per thing actually done, so the state above can be read
without reconstructing it from SLURM history.

- **2026-07-29** — Live search-quality diagnostics landed (T5): root prior
  entropy, visit entropy and visit concentration now logged per level per
  episode. `pytest tests/` = 85 passed. Note bare `pytest` collects `.venv/`
  and dies with 113 collection errors — always scope it to `tests/`.
- **2026-07-29** — T1 eval sweeps submitted: jobs **5825183** (Level1-2) and
  **5825184** (Level1-1), both off `checkpoints/best.pt`, 3h wall, 1 GPU each.
  Results pending. T1 code committed the same day in `684e829`.
- **2026-07-27** — Project paused for VGDL. Nothing MuZero left queued.
- **2026-07-21** — `level1-2-diag-v1` finished: 15M env / 763K train steps,
  peak self-play completion **0.68** @ step 698K, greedy replay 0 throughout.
- **2026-07-20** — Sims sweep (job 5727108) on Level1-1: 50/200/400 sims × 3
  seeds gave bit-identical returns per sim count (154.70 / 154.80 / 154.00),
  never completing. First hard evidence that more search does not help.
  `scripts/submit_curriculum.sh` written (committed in `ed4808d`, never run).
- **2026-07-18** — `level1-1-diag-v1` finished: 15M env / 780K train steps,
  peak self-play completion **0.84** @ step 696K, greedy replay 0 throughout.
  Discount fix (0.997 → 0.999 + `completion_bonus` 200) confirmed as what
  unlocked completions at all — every pre-fix run was ≤4%.
- **2026-07-16** — `imit-all-v1` and `imit-sub01-v1` each completed their 10M
  env-step budget (382K / 398K train steps). No `best.json` was written for
  either, i.e. neither ever set a rolling-completion-rate high.
