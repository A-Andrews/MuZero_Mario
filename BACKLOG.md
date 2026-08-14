# MuZero-Mario backlog

**Status (2026-08-14): T6 fleet #1 was killed by the home storage quota ~2h15
in; storage is migrated to `/projects`, the failure modes are patched, and
fleet #2 is queued (jobs 6017303-6017326) resuming from the surviving
checkpoints. All four diagnostic jobs (T1 follow-ups + bench-2gpu) completed
and are read — results folded in below.**

- **T6 fleet #1 died 2026-08-12** — all 12 leg-1 jobs (5990729-5990751, odd)
  exited 1 simultaneously at their next checkpoint-save boundary: 12 × 271 MB
  every 16 min blew the **hard 101 GB home quota** (no soft limit, no grace).
  The `.err` files were empty because wandb's console redirect buffered the
  tracebacks. ~40K train steps / ~600-860K env steps per run survived on disk.
- **Fixed 2026-08-14**: `outputs/` is now a symlink to
  `/projects/u6oz/atdandrews/MuZero_Mario/outputs` (200 TB Lustre, no quota);
  failed checkpoint saves warn-and-continue instead of killing the run (and
  clean up their `.tmp`); `training.checkpoint_keep` makes rotation
  configurable; submit scripts export `WANDB_CONSOLE=off` so the next crash
  lands in the `.err`. Home is back to 42 GB of 101 GB.
- **The greedy checkpoint scans landed the verdict** (see "The open problem"):
  greedy completion is a **knife-edge property of individual checkpoints**,
  not of the trained policy — and on Level1-2 `best.pt` is greedily *worse*
  than the final checkpoint. `best.pt` selection is the weak link for T8.
- **`bench-2gpu` is clean**: 200K env steps in 30 min (~111 env-steps/s) on
  the 2-GPU split. T7 is unblocked on this axis.

Next actions, in order:
0. **Sanity-check the first fleet-#2 specialist that starts**: it must resume
   from `latest.pt` (~step 40K) rather than start fresh, and
   `selfplay/root_exploration_eps` should continue its 0.25 → 0.05 decay.
   Fleet #1 already validated the fresh-start path (eps anneal + stochastic
   starts ran 2h15 in production without incident; spec-level3-2 hit 0.14
   completion by step 42K, ahead of the diag pace).
1. **Decide whether T7's arms get the eps anneal** before launching them;
   `submit_curriculum.sh` is still on constant eps=0.25 by decision (T2.1).
2. **Rethink `best.pt` selection before T8** — the scans show the
   noisy-self-play rolling rate can pick a checkpoint that deadlocks greedily
   (Level1-2) while a later one completes.

---

## Where things stand

| Run | Steps | Peak self-play completion | Greedy replay | Status |
|---|---|---|---|---|
| `level1-1-diag-v1` | 15M / 780K train | **0.84** @ step 696K | 0/15 in-training evals; **completes off `best.pt`** (T1) | done 2026-07-18 |
| `level1-2-diag-v1` | 15M / 763K train | **0.68** @ step 698K | **1/15** in-training evals (step 501K) | done 2026-07-21 |
| T1 eval sweep Level1-2 | — | — | — | **done**, job 5825183, 18/18 cells |
| T1 eval sweep Level1-1 | — | — | — | **done**, job 5825184, TIMEOUT, 14/18 cells |
| T1 fill-in (1-1, `pb_c`=2.5) | — | — | — | **done 2026-08-12**, job 5989666: best cell 0.80 @ eps=0.1/temp=0.1 |
| Greedy checkpoint scans | — | — | — | **done 2026-08-12**, jobs 5989667 (1-1: greedy completes at 2/11 ckpts) / 5989669 (1-2: last 3 ckpts complete, `best.pt` deadlocks) |
| T6 fleet #1 (12 levels) | ~40K train / 600-860K env each | 0.14 (3-2), 0.02 (1-1), 0.01 (2-2), 0 others | — | **killed 2026-08-12 by home quota**, jobs 5990729-5990752; checkpoints survive |
| T6 fleet #2 (12 levels) | — | — | — | **queued 2026-08-14**, jobs 6017303-6017326 (12 × 2 legs), resumes fleet #1's checkpoints |
| `bench-2gpu` (T4 benchmark leg) | 200K env / 8K train | — | — | **done 2026-08-12**, job 5989959: 30 min, ~111 env-steps/s on the 2-GPU split |
| `curriculum-v1` (12 levels, 60M) | — | — | — | **never submitted**; unblocked, pending the bench leg + the T7 eps decision |

The discount fix (0.997 → 0.999 + `completion_bonus` 200) is confirmed and is
what unlocked completions at all; every pre-fix run was ≤4%.

### The open problem (**rewritten 2026-08-11 — T1 refuted half of it**)

T1 says the answer is **level-dependent**, and it also found the metric that
posed the question is a much weaker instrument than assumed.

**On Level1-1 the premise is simply false.** Off `best.pt`, the fully-greedy
cell (eps=0, temp=0, `pb_c`=1.25) *completes the level*, and every perturbation
degrades it monotonically — noise and temperature alike. Nothing is
load-bearing here except the search itself.

**On Level1-2 the premise holds.** Greedy is 0 at both `pb_c` values, and at
temp 0 the completion rate climbs with noise alone (0 → 0.25 → 0.45 at
`pb_c`=1.25; 0 → 0.30 → 0.55 at 2.5). The greedy failure is a deadlock, not a
death: `pb_c`=1.25 greedy burns all 2000 steps stuck at `final_x`=482.

**Three corrections to what this file previously asserted:**

1. "Greedy replay 0 at every checkpoint" was wrong for Level1-2 — it logged
   `completed=1` at train step **501,156** (return 295.7, length 1800).
2. The in-training greedy eval runs on `training.replay_every_train_steps` =
   **50,000**, not `save_every` — so it is **15 single deterministic
   trajectories** across a 780K-step run, one 0/1 sample per ~50K steps. That
   is far too sparse to support "0 at every checkpoint" as a claim about the
   policy.
3. The Level1-1 contradiction is *not* a simulation-count artefact. Both diag
   runs were launched with `mcts.num_simulations=50` (see their
   `.hydra/overrides.yaml`), so the in-training eval used 50 sims too — the
   same as the sweep. What differs is the **weights**: `best.pt` is train step
   696,189, the nearest in-training eval was step 701,222 and hit the 2000-step
   cap without completing. 5K training steps apart, opposite outcomes.

**The scans answered this (2026-08-12, jobs 5989667/5989669): it is a
knife-edge.** Greedy completion is a property of individual checkpoints, not
of the trained policy:

- **Level1-1**: greedy completes at only **2 of 11** retained checkpoints
  (`best.pt` and step_760000, both reaching `final_x`=3266). The other nine
  stall at 1757-2978. "best.pt completes" (T1) is real but is a checkpoint
  lottery — 5K train steps separate completion from a 2000-step timeout.
- **Level1-2**: the inverse. Greedy completes at the **last three**
  checkpoints (755K/760K/763004) but **fails off `best.pt`**, deadlocked at
  `final_x`=482 for all 2000 steps. `best.pt` is selected on the *noisy*
  self-play rolling rate, and here that picked a checkpoint strictly worse
  greedily than the final one. **This is the T8 risk**: distillation teachers
  chosen by `best.pt` can be exactly the deadlocked weights.
- The 5989666 fill-in (Level1-1 `pb_c`=2.5 off `best.pt`, 20 eps/cell):
  best cell **0.80 [0.58,0.92]** at eps=0.1/temp=0.1; the training-noise
  setting eps=0.25/temp=0.25 is *worst* at 0.35. Mild noise beats both greedy
  and full training noise at this `pb_c`.

A related knife-edge was already visible in T1: Level1-1 greedy is 1.00 at
`pb_c`=1.25 and 0.00 at `pb_c`=2.5.

**What survives from the original diagnosis** (the reasoning below still holds,
and still explains Level1-2):

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
that knows the answer. **Scope (2026-08-11): T1 confirms this for Level1-2 and
refutes it for Level1-1**, where greedy off `best.pt` completes and noise only
costs. Both levels' policy heads are near-uniform by the CE numbers above, so
the near-uniform prior evidently is not by itself disqualifying — it is
survivable on a level with no hard chokepoint.

Level1-2's greedy failure is concrete: dies at x=850 (~25% in) into a cluster of
three Koopas, at frame ~250/283 of
`outputs/runs/level1-2-diag-v1/videos/step_750000/Level1-2.mp4`.

---

## T1 — Run the eval sweep (**done 2026-08-11 — results below**)

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

**Results (2026-08-11).** Job 5825183 (Level1-2) COMPLETED in 2:47:36, all 18
cells. Job 5825184 (Level1-1) **TIMEOUT at the 3h wall with 14/18 cells** —
benign, but the wall time was budgeted off Level1-2 episode lengths and the
Level1-1 greedy cells survive to the 2000-step cap, 3-4× longer. Missing cells
are all of `pb_c`=2.5 × eps=0.25, plus `pb_c`=2.5/eps=0.1/temp=0.25;
resubmitted as job **5989666**.

Completion rate, **Level1-1** at `pb_c_init`=1.25 (`best.pt`, step 696,189):

| eps \ temp | 0 | 0.1 | 0.25 |
|---|---|---|---|
| **0** | **1.00** (deterministic) | 0.70 | 0.35 |
| **0.1** | 0.80 | 0.65 | 0.55 |
| **0.25** | 0.70 | 0.50 | 0.50 |

Greedy completes; every perturbation degrades it, monotonically in both axes.
At `pb_c`=2.5 the greedy cell instead fails (2000-step cap, `final_x`=2914)
while the noisy cells sit at 0.45-0.80.

Completion rate, **Level1-2** at `pb_c_init`=1.25 (`best.pt`, step 698,817):

| eps \ temp | 0 | 0.1 | 0.25 |
|---|---|---|---|
| **0** | **0.00** (deterministic) | 0.40 | 0.30 |
| **0.1** | 0.25 | 0.45 | 0.35 |
| **0.25** | 0.45 | 0.30 | 0.25 |

Greedy is 0 at both `pb_c` values. `pb_c`=2.5 is mildly better overall (best
cells 0.55) but does **not** rescue the greedy cell.

**Two caveats when reading `summary.csv`.** Deterministic cells are n=1: the
env has no stochasticity, so 1.00 there means "this one trajectory finishes",
which is exact rather than an estimate — but the `ci95_lo`/`ci95_hi` columns on
those rows are Wilson intervals on n=1 (0.2065-1.0) and are meaningless. Do not
quote them. And single-trajectory cells are knife-edges: 1-1 greedy flips from
1.00 to 0.00 on `pb_c_init` alone.

**Against the decision table below**: outcome 1 (noise is a crutch) holds for
Level1-2 only; outcome 2 (`pb_c`=2.5 rescues greedy) is refuted on both levels;
outcome 3 does not apply. So **T2.1 (eps anneal) is justified, but its value is
level-dependent** — it is the fix for the Level1-2 chokepoint and is close to a
no-op cost on Level1-1. That matters for T6/T8, where a single global eps
schedule is applied to all 12 specialists.

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
   schedule — **done 2026-08-11**. `mcts.root_exploration_eps_schedule`,
   `[[train_step, eps], ...]` knots linearly interpolated, `null` (default)
   keeps the old constant eps. The mechanism is `src/muzero/schedules.py`,
   extracted from the linear interpolation that already lived inside
   `MixedBuffer` so the mix ratio and the eps anneal share one implementation;
   the worker mutates `mcts.root_exploration_eps` per step, which MCTS reads at
   `run()` time. Effective value logged as `selfplay/root_exploration_eps`.
   `submit_specialist.sh` ships `[[0,0.25],[500000,0.05]]`, bottoming out where
   its temperature schedule does.

   **`submit_curriculum.sh` was deliberately left unchanged**, so T7 still runs
   constant eps=0.25 unless the schedule is passed explicitly:
   `'mcts.root_exploration_eps_schedule=[[0,0.25],[3000000,0.05]]'` is the grid
   matching its temperature schedule. Decide this before launching T7 — it is
   the one knob that differs between the arms otherwise.
2. **Raise self-play `mcts.num_simulations`** 50 → 100+. Sharper visit
   distributions are the direct fix for a policy head sitting near uniform —
   the training *target* is currently too diffuse. Costs throughput; the
   untested 2-GPU split is the headroom.
3. **Stochastic starts** — **done 2026-08-11**, and promoted to first because
   T1's whole interpretive mess (n=1 cells, meaningless Wilson CIs on them,
   outcomes flipping on `pb_c_init` alone) traces to the deterministic env.
   `env.noop_max=30` + `env.skip_to_control=true` in `conf/env/mario.yaml`.

   Two things found while building it, both of which would have made a naive
   implementation a silent no-op:
   - **Every level opens with a scripted intro**, not just Level1-2 — 123
     frames on Level1-1, 117 on Level1-2 (`player_state` 0 → 7 → 8). A 0-30
     frame delay lands entirely inside it and is absorbed, so `noop_max`
     alone changes nothing: verified `final_x` bit-identical across 6 seeds.
     Hence `skip_to_control`, which advances to `player_state == 8` first.
   - **`env.reset()` returns an empty info dict**, so `player_state` can only
     be read by stepping; the skip has to be a do-while. Both are covered by
     `tests/test_stochastic_start.py` (stub emulator, no ROM needed).

   Verified against the real emulator: holding right+B for 400 agent steps
   gives 1 distinct outcome across 6 seeds with the knobs off, and 3 (1-1) /
   5 (1-2) with them on, Level1-2 `final_x` spreading 184-191.

   **Consequences to carry forward.** Episodes lose ~30 agent steps of
   uncontrollable title card, which shifts the autocurriculum's inverse-length
   weighting and makes episode lengths incomparable to the diag runs.
   `run_replay_rollout` defaults both knobs **off**, so greedy eval stays
   reproducible and comparable to the 0.84/0.68 baselines — evaluate the new
   runs both ways.

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

## T4 — The curriculum run (unblocked; benchmark leg queued)

`scripts/submit_curriculum.sh` was written 2026-07-20 and committed in `ed4808d`
but **still never submitted**. 12 levels, 60M env steps, 4 chained 24h legs,
2 GPUs, 64 workers, 1M buffer.

The benchmark leg is queued as job **5989959** — the 2-GPU split and the
64-worker throughput have never run in production, and T7 commits ~384 GPU-h
to them. It also now exercises the T2.3 env, since SLURM reads the working
tree at run time:

```bash
bash scripts/submit_curriculum.sh bench-2gpu 1 training.total_env_steps=200_000
```

**Two things to settle before the real chain goes out.** The `eps=0.25`-forever
objection is now fixable but *not fixed* — `submit_curriculum.sh` was left
unchanged by decision, so the schedule must be passed explicitly (see T2.1).
The 50-sims objection is withdrawn: T1 found `pb_c_init`, not simulation count,
is what flips outcomes, and the July 20 sweep found 50/200/400 bit-identical.

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

# The experiment program (T6-T8)

Planned 2026-07-29. Three arms, run in this order because each feeds the next:

| Arm | What | Produces | Depends on |
|---|---|---|---|
| **T6** | 12 per-level specialists | the distillation teachers | T1 + T2 |
| **T7** | autocurriculum × {no human, human} | the generalist baselines | T1 + T2 |
| **T8** | distil the 12 specialists into one all-level model | the headline model | T6 |

T7 and T6 are independent of each other and can run concurrently if the
cluster has the GPUs; T8 cannot start until T6's checkpoints exist.

### Why all three blocked on T1/T2 (**resolved 2026-08-11**)

Decided 2026-07-29: do **not** launch on the then-current recipe. Root
Dirichlet noise, not the policy head, was carrying the 0.59-0.84 self-play
completion rate. Survivable for a specialist you only ever evaluate with noise
on, fatal for **T8** — the teachers' stored visit distributions are what the
student imitates, and a near-uniform teacher distils into a near-uniform
student. Launching before T2 risked spending ~1,000 GPU-hours measuring the
noise artefact.

**Discharged.** T1 read (level-dependent: noise load-bearing on 1-2, pure cost
on 1-1), T2.1 landed so the teachers' eps decays to 0.05 rather than sitting at
0.25 forever, T2.3 landed so the numbers are estimates rather than single
trajectories. T6 launched on that basis.

**T7 is the remaining exposure**: `submit_curriculum.sh` still runs constant
eps=0.25 by decision, so if it launches as-is its arms differ from T6's
specialists on exactly the knob this section says matters. Either pass
`'mcts.root_exploration_eps_schedule=[[0,0.25],[3000000,0.05]]'` or record the
mismatch deliberately — do not let it happen by default.

### Shared decisions across all three arms

- **Recipe**: the `submit_curriculum.sh` header recipe (discount 0.999,
  `completion_bonus` 200, `muzero_mario_medium`, cosine LR to the 1e-4 floor),
  plus whatever T2 lands. Every arm uses the same one so the comparison is
  clean.
- **Evaluation**: score every arm with the `eval_sweep.py` grid, **not** with
  greedy `replay/<level>_completed`. Reinforced by T1 (2026-08-11): that metric
  is **one deterministic trajectory every 50K train steps** — 15 samples per
  run — so it is a knife-edge probe, not an estimator, and it disagreed with a
  matched-settings sweep off `best.pt` on Level1-1. Report completion rates
  with Wilson 95% intervals, at matched eps/temperature across arms, and note
  that the deterministic cells are exact-but-n=1 (ignore their CI columns).
- **Model size**: student and specialists all `muzero_mario_medium` (192ch/10).
  Keeps T8 an honest test of whether one net of *fixed* capacity absorbs 12
  specialists, rather than a capacity story. `muzero_atari` is the fallback only
  if the student underfits, and CLAUDE.md flags it at ~5 env-steps/s.

### Compute estimate

Measured from `level1-2-diag-v1`: jobs 5727109 (24h, TIMEOUT) + 5727110
(12:25) = **~36 GPU-hours for 15M env steps** on 1 GPU.

| Arm | Runs | Cost each | Total |
|---|---|---|---|
| T6 specialists | 12 | ~36 GPU-h (2 chained legs, 1 GPU) | ~440 GPU-h |
| T7 curriculum | 2 | ~192 GPU-h (4 legs × 24h × 2 GPUs) | ~384 GPU-h |
| T8 dump + student | 1 | ~18 GPU-h dump + ~192 GPU-h | ~210 GPU-h |

**~1,030 GPU-hours ≈ 43 GPU-days.** Wall-clock is far shorter — the QOS allows
256 concurrent jobs, so all 12 specialists run in parallel (~2 days wall).

---

## T6 — The specialist fleet (**fleet #2 queued 2026-08-14**)

12 single-level runs, 15M env steps each, matching the diag-run budget so the
results are comparable to `level1-1-diag-v1` (0.84) and `level1-2-diag-v1`
(0.68). Run dirs `outputs/runs/spec-<level>/`.

**Fleet #1** (jobs 5990729-5990752, launched 2026-08-11) ran 2h15 and was
killed fleet-wide by the hard 101 GB home quota — 12 × 271 MB checkpoints
every 16 min. Every leg-1 died at its next save boundary; every leg-2 was
cancelled on the dependency. The ~40K-train-step checkpoints survived, and
the 2h15 of production self-play did validate both new mechanisms (eps anneal,
stochastic starts) and showed spec-level3-2 at 0.14 completion by step 42K —
ahead of the diag pace. **Fleet #2** (jobs **6017303-6017326**, queued
2026-08-14) resumes those checkpoints via the normal `latest.pt` path, with
storage now on `/projects` and failed saves made non-fatal.

The blocker is cleared: `submit_all_levels.sh` submitted *unchained*
`submit_single_level.sh` jobs that set no `run_name` and so could not
auto-resume — at 24h wall against ~36h of work per level, every run would have
died two-thirds finished. `scripts/submit_specialist.sh` now wraps
`submit_chain.sh` with `env.levels=[<level>]`, `autocurriculum.enabled=false`
and `run_name=spec-<level>`.

```bash
for L in Level1-1 Level1-2 Level1-3 Level2-1 Level2-2 Level2-3 \
         Level3-1 Level3-2 Level3-3 Level4-1 Level4-2 Level4-3; do
    bash scripts/submit_specialist.sh "$L" 2
done
```

**Two deliberate departures from the diag recipe**, both from T1 — record them
when reading the results:
- **Stochastic starts** (T2.3) are on, and the diag runs had none. Completion
  rates stay comparable; **episode lengths do not** (~30 agent steps of title
  card left the trajectory).
- **The eps anneal** (T2.1) decays 0.25 → 0.05 by train step 500K. The diag
  runs held eps at 0.25 throughout, so a specialist that matches 0.84 is
  strictly the stronger result — it did it on a decayed noise budget.

Success criterion: each specialist's own-level completion rate under the T1
eval grid. Expect a wide spread — Level1-1 hit 0.84 while Level1-2's greedy
policy dies at x=850, and 4-x are unattempted.

## T7 — Autocurriculum ± human teacher (blocked on T1/T2)

Two runs off `submit_curriculum.sh`, identical except for the `imitation:`
block:

```bash
bash scripts/submit_curriculum.sh curriculum-nohuman 4
bash scripts/submit_curriculum.sh curriculum-human 4 \
    imitation.enabled=true \
    imitation.pretrain_steps=50_000 \
    'imitation.mix_ratio_schedule=[[0,0.25],[700_000,0.0]]'
```

Chosen 2026-07-29: **pretrain + annealed mix**, all subjects. The anneal is the
point — a one-hot BC anchor that never fades drags on the policy indefinitely,
which is what `mix_ratio_schedule` was added for. Schedule thresholds are RL
train steps and the 50K pretrain offset is applied automatically.

**Run the benchmark leg first.** The 2-GPU split and 64-worker throughput have
never run in production:
`bash scripts/submit_curriculum.sh bench-2gpu 1 training.total_env_steps=200_000`.

**Caveat — the human arm covers 11 of 12 levels.** Verified against the corpus
2026-07-29: there is no `level-w2l2`, i.e. **Level2-2 has zero human data**
(humans also never played w7l2 or the castle -4 levels, but those are not in
`env.levels`). The autocurriculum will keep sampling Level2-2 and the human
buffer contributes nothing there, so any T7 win must be checked per level
before it is attributed to the human teacher. `levels: match_env` filters by
filename tag and will simply find no w2l2 files — confirm the loader tolerates
a requested level with zero matches rather than raising.

## T8 — Distillation into one all-level model (blocked on T6)

Goal: turn the 12 T6 specialists into a single model that plays all 12 levels.
Mechanism chosen 2026-07-29: **teacher corpus → BC pretrain → RL with an
annealed mix** — i.e. reuse the imitation pipeline end to end, with a
specialist-generated corpus in place of the human one. A specialist is just
another teacher.

**Why this and not an online KL loss**: it needs almost no new machinery. The
pinned never-evicted buffer, `MixedBuffer`, `mix_ratio_schedule`, the BC
accuracy/CE metrics and the whole targets/reanalyze path already exist and are
corpus-agnostic. No teacher has to stay resident in GPU memory. The tradeoff is
a fixed corpus: the student never gets teacher labels on the states *it*
visits, so compounding error is uncorrected — if the student plateaus well
below its teachers, that is the first thing to suspect, and the escalation is
the online-KL variant.

**Step 1 — the dumper (new: `scripts/dump_specialist_trajectories.py`).**
Load each specialist checkpoint, roll out its own level, emit `Trajectory` .npz
in the *same format and naming convention* as `convert_human_bk2.py`. Two
details make this free:
- `human_data.py` filters on filename only — subject by `<subject>_` prefix,
  level by a `level-wXlY` tag. Name the files
  `spec-w1l1_ses-000_task-mario_level-w1l1_rep-000_seg0.npz` and both filters
  work with **zero loader changes**; the specialist even shows up as its own
  selectable "subject".
- Store the **raw MCTS visit distribution** in `policies`, not a one-hot. This
  is strictly richer supervision than the human corpus (whose one-hot is what
  makes its policy loss pure BC) and the loader does not care.

**Step 2 — the student.**

```bash
bash scripts/submit_curriculum.sh distilled-v1 4 \
    imitation.enabled=true \
    imitation.data_dir=outputs/specialist_trajectories \
    imitation.pretrain_steps=50_000 \
    imitation.max_transitions=1_500_000 \
    'imitation.mix_ratio_schedule=[[0,0.25],[700_000,0.0]]'
```

**Open question the dumper cannot dodge — what sampling policy to dump with.**
If T1 confirms noise is load-bearing, then a noise-free dump yields a corpus of
*failures* (greedy completes 0), while a noisy dump yields successful
trajectories whose stored visit distributions have Dirichlet noise baked into
the thing the student is asked to imitate. Neither is right. This is the
strongest reason T8 waits for T2's eps anneal: a specialist trained to work at
eps≈0.05 can be dumped near-greedy and still succeed. Decide the dump
temperature from the T1 grid, and record it — it is the single most
consequential knob in this arm.

**RAM ceiling — cap the corpus.** Obs are 36.9 KB/step and the pinned buffer is
resident. The human corpus reference point is 1.53M steps ≈ 56.5 GB against a
110 GB single-GPU share. So `max_transitions=1_500_000` ≈ 55 GB, i.e. roughly
**125K steps ≈ 300 episodes per level** — coincidentally about the human
corpus's ~300 segments/level. Dumping more than that just gets evicted at load
time; dump to the cap, not beyond it.

**Comparisons that make T8 interpretable**, all on the same eval grid:
1. student vs. each specialist **on that specialist's own level** — does one net
   hold 12 skills, or does it average them away?
2. student vs. `curriculum-nohuman` (T7) — is distillation better than just
   training the generalist directly?
3. student vs. `curriculum-human` (T7) — specialist teachers vs. human teachers,
   the same pipeline with different corpora. This is the cleanest scientific
   comparison in the whole program and is the reason to keep both arms on
   identical imitation settings.

---

## Incidental findings worth keeping

- **Every level has a scripted intro, not just Level1-2** (corrected
  2026-08-11 by direct measurement). Level1-1 is 123 frames ≈ 31 agent steps,
  Level1-2 117 frames ≈ 29, `player_state` running 0 → 7 → 8 ("in control").
  Input is ignored throughout and `player_x_pos` reads 0, so every episode in
  every run to date opened with ~30 uncontrollable agent steps. Relevant when
  reading episode lengths and setting `max_steps`; `env.skip_to_control`
  (T2.3) now removes them.
- The login node kills multi-threaded torch (`libgomp: Thread creation failed`).
  Anything beyond a trivial single-threaded script needs `salloc`/`sbatch`;
  `OMP_NUM_THREADS=1` is enough for quick import-level checks.
- **`pytest tests/` (full suite) dies silently on the login node** at
  `test_inference_server.py::test_initial_inference_matches_direct_net` —
  test #38 of 110, exit 1, no traceback, and it dumps a multi-GB core file
  into the repo. Reproducible on an unmodified tree; the same test **passes
  standalone**, and the whole suite passed on 2026-08-11 — so it's a
  login-node environment/ordering problem (probably the same thread-creation
  class as the libgomp finding above), not a code regression. Run the full
  suite inside an allocation; scoped files are fine on the login node.
  (Found 2026-08-14; the 6.2 GB core it dumped was deleted.)

---

## Activity log

Newest first. One line per thing actually done, so the state above can be read
without reconstructing it from SLURM history.

- **2026-08-14** — **T6 fleet #2 queued**: jobs **6017303-6017326** (12 × 2
  legs via `submit_specialist.sh`, identical recipe), resuming fleet #1's
  ~40K-step checkpoints through the normal `latest.pt` auto-resume path.
- **2026-08-14** — **Quota postmortem + storage migration.** Fleet #1's death
  diagnosed as the hard 101 GB home quota (all 12 legs died at their next
  16-min save boundary; empty `.err`s because wandb buffered the console).
  Fixes: `outputs/` (39 GB) moved to
  `/projects/u6oz/atdandrews/MuZero_Mario/outputs` (200 TB Lustre, no quota)
  with a symlink at the old path (rsync verified zero-diff before deleting
  the home copy; home 82 → 42 GB, incl. a 6.2 GB core dump); checkpoint saves
  now warn-and-continue on `OSError` and clean up partial `.tmp`s
  (regression-tested); `training.checkpoint_keep` added (default 10);
  `WANDB_CONSOLE=off` exported by all three submit scripts; `.gitignore`
  `outputs/` → `outputs` (trailing slash doesn't match symlinks). Found in
  passing: full `pytest tests/` dies silently mid-suite on the login node
  (see incidental findings) — checkpoint tests 12/12, crashing test passes
  standalone, pre-existing on a clean tree.
- **2026-08-12** — **T6 fleet #1 died** ~2h15 in (see postmortem above). The
  same morning the four diagnostics completed clean: **5989959**
  (`bench-2gpu`: 200K env steps / 30 min ≈ 111 env-steps/s on the 2-GPU
  split), **5989666** (Level1-1 `pb_c`=2.5 fill-in: best 0.80 @
  eps=0.1/temp=0.1, training-noise cell worst), **5989667/5989669** (greedy
  checkpoint scans: greedy completion is a per-checkpoint knife-edge; 1-1
  completes at 2/11 ckpts, 1-2 at the last 3 but *not* `best.pt`, which
  deadlocks at x=482 — the T8 teacher-selection risk).
- **2026-08-11** — VGDL finished; its two stranded jobs (5843779 s3-sync,
  5843780 readme, both `afterok` behind the failed 5843608 and therefore
  unrunnable) cancelled. Queue is now MuZero-only.
- **2026-08-11** — **T6 launched**: 12 specialists × 2 chained legs (jobs
  5990729-5990752) via `submit_specialist.sh`, carrying T2.1 + T2.3. Smoke
  test (full pipeline, CPU, both new knobs on) passed first. VGDL is finished,
  so MuZero has priority again. T7 still held on the `bench-2gpu` result and
  the decision about whether its arms get the eps anneal.
- **2026-08-11** — **T2.1 root-Dirichlet anneal landed**
  (`mcts.root_exploration_eps_schedule`, null by default). Shared
  piecewise-linear helper `src/muzero/schedules.py` extracted from
  `MixedBuffer` rather than duplicated; effective eps logged as
  `selfplay/root_exploration_eps`. Wired into `submit_specialist.sh` only —
  `submit_curriculum.sh` left on constant eps by decision, so T7 needs the
  override passed explicitly. `pytest tests/` = 109 passed.
- **2026-08-11** — **T2.3 stochastic starts landed** (`env.noop_max=30`,
  `env.skip_to_control=true`). Found that *every* level has a ~120-frame
  scripted intro and that `env.reset()` returns an empty info dict — either
  one alone would have made the feature a silent no-op. Verified against the
  real emulator (deterministic: 1 outcome / 6 seeds; on: 3-5). `pytest tests/`
  = 100 passed. `submit_specialist.sh` written and dry-run verified, clearing
  T6's code blocker; `bench-2gpu` submitted as job **5989959**.
- **2026-08-11** — Follow-ups queued: **5989666** (Level1-1 fill-in for the 4
  cells 5825184 lost to the wall, 4h), **5989667**/**5989669** (greedy
  checkpoint scans on 1-1/1-2 via the new
  `scripts/submit_greedy_ckpt_scan.sh`). The scans exist because the
  greedy-completes-0 premise rests on only 15 single deterministic
  trajectories per run, and `best.pt` (step 696,189) completes Level1-1
  greedily while the in-training eval 5K steps later did not.
- **2026-08-11** — **T1 results read.** Answer is level-dependent: greedy
  completes Level1-1 and every perturbation hurts; greedy is 0 on Level1-2 and
  noise alone recovers 0.45-0.55. Job 5825183 clean (18/18 cells), 5825184
  TIMEOUT (14/18). Also corrected two long-standing claims in this file —
  Level1-2 greedy replay was **not** 0 at every checkpoint (completed=1 at
  step 501,156), and the in-training greedy eval runs every
  `replay_every_train_steps`=50K, i.e. 15 samples per run, not per checkpoint.
  The sims-mismatch hypothesis was checked and **ruled out**: both diag runs
  used `mcts.num_simulations=50`, matching the sweep.
- **2026-07-29** — Experiment program T6-T8 planned (specialists → curriculum
  ±human → distillation). All three gated on T1/T2 by decision, because
  distilling a noise-dependent teacher would poison T8. ~1,030 GPU-h estimated.
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
