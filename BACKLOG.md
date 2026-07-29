# MuZero-Mario backlog

**Status: T1 running as of 2026-07-29; the T6-T8 experiment program is planned
and gated on it.** The two eval sweeps are queued on SLURM (jobs **5825183**
Level1-2, **5825184** Level1-1, 3h wall each). VGDL still holds priority for
everything else. Both diagnostic runs finished cleanly; no work is at risk.

Next action: **read the T1 results** when the jobs land, then pick T2 or T3 by
the decision table under T1. That unblocks T6-T8, the ~1,030 GPU-hour program
at the bottom of this file.

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

# The experiment program (T6-T8)

Planned 2026-07-29. Three arms, run in this order because each feeds the next:

| Arm | What | Produces | Depends on |
|---|---|---|---|
| **T6** | 12 per-level specialists | the distillation teachers | T1 + T2 |
| **T7** | autocurriculum × {no human, human} | the generalist baselines | T1 + T2 |
| **T8** | distil the 12 specialists into one all-level model | the headline model | T6 |

T7 and T6 are independent of each other and can run concurrently if the
cluster has the GPUs; T8 cannot start until T6's checkpoints exist.

### Why all three block on T1/T2

Decided 2026-07-29: do **not** launch on the current recipe. The open problem
above says root Dirichlet noise, not the policy head, is carrying the 0.59-0.84
self-play completion rate. That is survivable for a specialist you only ever
evaluate with noise on, but it is fatal for **T8** — the teachers' stored visit
distributions are what the student imitates, and a near-uniform teacher policy
distils into a near-uniform student policy. Distilling before T2 lands risks
spending the whole ~1,000 GPU-hour program measuring the noise artefact.

So: read T1 (jobs 5825183/5825184, already queued), apply the T2 fixes it
justifies — the eps anneal (T2.1) is the one that matters here, since it is what
forces the policy head to stand on its own — then launch T6/T7.

### Shared decisions across all three arms

- **Recipe**: the `submit_curriculum.sh` header recipe (discount 0.999,
  `completion_bonus` 200, `muzero_mario_medium`, cosine LR to the 1e-4 floor),
  plus whatever T2 lands. Every arm uses the same one so the comparison is
  clean.
- **Evaluation**: score every arm with the `eval_sweep.py` grid, **not** with
  greedy `replay/<level>_completed`. That metric reads 0 on models with 0.84
  self-play completion, so it cannot rank these arms. Report completion rates
  with Wilson 95% intervals, at matched eps/temperature across arms.
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

## T6 — The specialist fleet (blocked on T1/T2)

12 single-level runs, 15M env steps each, matching the diag-run budget so the
results are directly comparable to `level1-1-diag-v1` (0.84) and
`level1-2-diag-v1` (0.68).

**Blocker — `submit_all_levels.sh` cannot do this as written.** It submits one
*unchained* `submit_single_level.sh` job per level, and that script sets no
`run_name`, so it cannot auto-resume. At 24h wall and ~36h of work per level,
every run would die two-thirds finished with no resume path. Needs a chained
per-level submitter — `submit_chain.sh` already has the mechanism, but it calls
`submit_autocurriculum.sh`; the per-level variant needs `env.levels=[<level>]`,
`autocurriculum.enabled=false` and `run_name=spec-<level>`.

```bash
# after the new script exists:
for L in Level1-1 ... Level4-3; do
    bash scripts/submit_specialist.sh "$L" 2 training.total_env_steps=15_000_000
done
```

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
