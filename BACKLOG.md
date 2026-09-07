# MuZero-Mario backlog

**Status (2026-09-05) — PROJECT PAUSED (internship). All three training tracks
are finished; nothing is mid-experiment.**

- **23 levels have a trained model**, covering **all 22 levels with CNeuroMod
  human/brain data**, plus Level2-2 (a model, but humans never played it).
  17 pure self-play, 5 imitation-rescued (1-3, 4-3, 5-2, 7-3, 8-2), and
  Level5-3 which **nothing solves** — 15M env steps of self-play and 15M with
  imitation, zero completions, dying at a fixed wall (x~907) every rollout.
- **T7 answered its question.** Both curriculum arms reached the full 60M
  env-step budget: `curriculum-human` 0.11, `curriculum-nohuman` 0.02, with the
  no-human arm flat from 23.9M onward. The human teacher is what let the
  12-level curriculum learn at all. Pooled over 12 levels, so still owed a
  per-level read (Level2-2 is the zero-human-data control).
- **The headline caveat is unchanged and matters for anyone using these
  models:** completion rates are measured with MCTS root-Dirichlet noise on.
  Greedy is far weaker — 8 of 19 levels finished at all in the 2026-09-03
  benchmark. A model at 0.9 self-play may still never finish the level greedily.
- **T8 has never been run.** Its dumper is written and it is the natural next
  experiment.

**Both closing jobs landed before the pause:**
- job 6338095 — greedy benchmark across the 22 levels with a `best.pt`:
  **10 of 22 finish greedily**, against 5/12 for the original fleet. Best are
  1-3, 3-3 and 6-1 at 5/5; the rescues 5-2 (3/5) and 7-3 (2/5) finish greedily
  despite low self-play rates, while 8-2 — the best rescue at 0.52 self-play —
  is 0/5 and therefore entirely noise-dependent. Figure and JSON are committed.
- job 6338250 — **the deliverable**:
  `/projects/u6oz/atdandrews/MuZero_Mario/exports/muzero_mario_models_20260905.zip`,
  23 checkpoints, **1123 MB**, manifest at sha e8a962b. Level5-3 is correctly
  marked `completed_level: false` / `latest.pt`.

**A new thread opened 2026-09-07 (T10, lesion study)** — machinery built, a
pilot and a simulation sweep run. Two results: the policy head's **1,266**
parameters matter more than the forward model's **7.0M**, and a broken value
head gets *worse the more the agent searches* (1.00 -> 0.50 completion from
10 to 200 simulations) while the intact model sits at ceiling throughout.

**Where to pick this up:** the numbered list below. Item 1 is the one that
blocks someone else's work, not just ours.

- **T6 fleet #2 finished 2026-08-16** — jobs 6017303-6017326, all reached
  `TRAINING_COMPLETE`. Final rates at eps=0.05: 3-2 0.89, 3-3 0.87, 1-1 0.72,
  3-1 0.40, 4-1 0.37, 1-2 0.33, 2-3 0.32, 2-1 0.31, 2-2 0.24, 4-2 0.08,
  and 1-3 / 4-3 at **0.00**.
- **The two dead levels were rescued with human demos (2026-08-20)**, runs
  `spec-imit-level1-3` / `spec-imit-level4-3` via
  `submit_specialist_imitation.sh` (imitation + completion-gated eps anneal).
  **Level1-3 escaped outright**: first completion at RL step 249k against
  *never* in ~97k T6 episodes, best rolling rate **0.85**, root Q 27.4 -> 81.4,
  and greedy replay completes at the annealed eps floor. **Level4-3 only
  cracked open**: first completion at 79% of budget, best rate 0.01, ended
  0.00. The mechanism works where the demos beat the wall in time.
- **Human comparison is now automatic** (`human_compare:`, commit ab0ceab).
  Every checkpoint replay is scored against a human on the same level. Two
  scale bugs were found and fixed doing it: the comparator must be the human
  **first-life** rate (agent episodes are `done_on_life_loss`, so the per-rep
  rate overstates humans by 8-16 points), and `run_replay_rollout` never
  passed the run's `completion_bonus`, so `replay/<level>_return` was on a
  different scale from `selfplay/episode_return` in **every run to date**.
- **The run-through benchmark (job 6083435) is the important new result.**
  `images/human_vs_agent_runthrough.pdf`: each level's `best.pt` played through
  its level 5x with stochastic starts. **5 of 12 levels finish at all; every
  level that finishes beats the human median time** (3-3 at 391 steps vs 830 is
  the widest margin; no level beats the human p10 — 3-3 comes closest at 391
  against a p10 of 375). But 7 of 12 never finish — including 1-1, which
  scores 0.72-0.91 in noisy self-play yet stalls at exactly x=2370 from five
  different starts. Level3-1 completed its *deterministic* training replay in
  811 steps and scores 0/5 here, so it is start-fragile, not merely
  noise-dependent.
- **T7 launched 2026-08-24** — `curriculum-nohuman` (jobs 6115540-6115543) and
  `curriculum-human` (6115544-6115548), 4 chained legs each. The open question
  from the previous status ("do T7's arms get the eps anneal?") is **decided:
  yes**, `[[0,0.25],[3000000,0.05]]`, bottoming out where the curriculum's
  temperature schedule does. Without it we would produce another set of
  checkpoints whose headline rate does not survive greedy eval — exactly what
  the benchmark just measured. The completion gate is left **off**: it keys on
  the run's first completion of *any* level, which an easy level unlocks almost
  immediately in a 12-level run.

Next actions, in order:

1. **Answered 2026-09-07: the collaborator feeds their own frames.** So the
   corpus frame-index gap **does not block them** and the conversion re-run is
   **not needed** for the collaboration — do not start it on their account.
   Their path needs only preprocessing parity, which the bundle handles:
   `load_model.frames_to_obs()`, verified byte-identical to the converter on a
   real .bk2 segment (992/992), and shipped in both bundles sent on 2026-09-07.

   **The one thing still worth confirming with them: what frame rate their
   source runs at.** `frames_to_obs` expects the emulator's native **60 Hz**,
   every frame, and consumes 4 per observation. Feeding it 30 fps video would
   silently produce observations spanning 8 emulator frames — the wrong
   temporal scale, no error raised, plausible-looking activations. This is the
   same class of silent failure the helper was written to prevent, one level up.

   The frame-index gap remains real as an **internal** limitation: if *we* ever
   want TR-aligned analysis from `outputs/human_trajectories/` ourselves, it
   still needs a frame counter emitted from `convert_human_bk2.py` and a re-run.
   Nothing is lost — .bk2 replay is frame-exact and deterministic.
2. **Done 2026-09-07 — both bundles sent.**
   `muzero_mario_models_20260905.zip` (23 levels, 1123 MB) and
   `muzero_mario_comparison_20260905.zip` (8 models, three matched pairs,
   396 MB), both from `/projects/u6oz/atdandrews/MuZero_Mario/exports/`.
   Note `muzero_mario_models_20260828.zip` (the earlier 12-level bundle) is
   also still out there, and its README predates `frames_to_obs`.
3. **Done 2026-09-05 — both T7 arms shipped** in
   `exports/muzero_mario_comparison_20260905.zip`, as `latest.pt` at exactly
   60.0M env steps each so they match on budget rather than on their individual
   bests. See the comparison-bundle section below for the other two pairs and
   the `imit-*` confound.
4. **T8 step 1 is done (2026-09-07, jobs 6382967-6382990) — the
   teacher-quality bar can now be decided from data, not guessed.** 21 of 23
   levels reported (Level1-2 / Level5-3 still running), **355 episodes /
   207,863 agent-steps** in `outputs/specialist_trajectories/`, at
   `--max-attempts-per-level 60`:

   | tier | levels | kept |
   |---|---|---|
   | at target | 1-1, 3-2, 6-1 | 40 each |
   | strong | 3-3 (34), 1-3 (33), 4-1 (33), 5-1 (23), 6-3 (22), 7-1 (18), 3-1 (17) | 17-34 |
   | thin | 2-1 (13), 8-2 (12), 2-2 (10), 2-3 (8), 5-2 (6), 7-3 (5), 4-2 (1) | 1-13 |
   | **empty** | **4-3, 6-2, 8-1, 8-3** | **0** |

   Four levels contribute nothing and three more contribute under 10 episodes,
   so a 23-level distillation target cannot be met from this corpus as it
   stands. Note **6-2 came out empty despite a 0.25 self-play rate** — worth a
   look before assuming the bar is just "rate too low". Also note the whole
   corpus is **207k agent-steps against the human corpus's 3.25M**, i.e. ~6%;
   raising `--episodes-per-level` on the levels that can supply them is the
   obvious next move.

5. **Then run T8 proper.** The imitation
   pipeline with the specialist corpus in place of the human one. Decide the
   **teacher-quality bar** first: the dumper filters to completing episodes, so
   levels whose specialist rarely completes even with noise (4-3 0.01, 8-1 0.01,
   8-3 0.03, 4-2 0.14) will be thin or empty; the script reports which came up
   short. T7's result is the argument for doing this — a mixed teacher stream
   demonstrably works.
6. **Re-run the T7 per-level read with n>1.** Done once at n=1 (see its
   section); the 8-of-12 return split and the Level2-2 control result both need
   several stochastic-start rollouts per level before they can be relied on.
7. **Finish T10.** Nearest-term: more `intact` rollouts per simulation count —
   the sweep's baseline is n=2 per cell and every other row is read against it.
   Then replicate the sweep on a second level (Level6-1 finishes 5/5 greedily,
   so it can show the same ceiling), and consider whether `reward` being free at
   every depth is a Mario artefact of dense shaped reward.
8. **Level5-3 is parked, deliberately.** 30M env steps across two recipes with
   zero completions. Do not chase it again without a new idea about the wall at
   x~907 — the failure is a death, not a timeout, so it is the same pit-gap
   archetype as T6's 1-3/4-3 but one the imitation rescue did not crack.

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
| T6 fleet #2 (12 levels) | 15M env each | 3-2 0.89, 3-3 0.87, 1-1 0.72, 3-1 0.40, 4-1 0.37, 1-2 0.33, 2-3 0.32, 2-1 0.31, 2-2 0.24, 4-2 0.08, **1-3 / 4-3 0.00** | 5/12 finish off `best.pt` (job 6083435) | **done 2026-08-16**, jobs 6017303-6017326 |
| `spec-imit-level1-3` | 15M env / 815K train | **0.85** @ step 553K | **completes** at the annealed eps floor | **done 2026-08-20**, job 6047664: rescued from 0.00 |
| `spec-imit-level4-3` | 15M env / 814K train | 0.01 @ step 660K | 0/5 | **done 2026-08-20**, job 6047668: cracked open, not solved |
| Run-through benchmark | — | — | — | **done 2026-08-24**, job 6083435 → `images/human_vs_agent_runthrough.pdf` |
| `bench-2gpu` (T4 benchmark leg) | 200K env / 8K train | — | — | **done 2026-08-12**, job 5989959: 30 min, ~111 env-steps/s on the 2-GPU split |
| `curriculum-v1` (12 levels, 60M) | — | — | — | **superseded** — launched as T7's two named arms instead |
| T9 fleet (11 levels, w5l1-w8l3) | 15M env each | 6-1 0.93, 5-1 0.73, 6-3 0.70, 7-1 0.48, 6-2 0.25, 8-3 0.03, 8-1 0.01, **5-2/5-3/7-3/8-2 0.00** | 3/7 finish off `best.pt` (job 6274769) | **done 2026-09-03**, jobs 6238199-6238220 |
| T9 rescue (4 dead levels) | 15M env each | 8-2 **0.52**, 5-2 0.25, 7-3 0.19, **5-3 0.00** | — | **done 2026-09-05**, jobs 6274783-6274790: 3 of 4 rescued; demo count did *not* predict success |
| `curriculum-nohuman` (T7 arm A) | 60M env / 2.70M train | **0.02** (flat from 23.9M) | — | **done 2026-09-05**, jobs 6115540-6115543 + 6249958-6249960 (ran out of legs at 39.4M, +3 legs) |
| `curriculum-human` (T7 arm B) | 60M env / 2.80M train | **0.11** — 5.5x arm A at equal budget | — | **done 2026-09-05**, jobs 6115544-6115548 + 6249961-6249963; BC pretrain 50K + mix anneal |

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

## T2 — Training changes (**T2.1 + T2.3 done; shipped in T6/T7**)

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

   **Resolved 2026-08-24**: `submit_curriculum.sh` is still unchanged (constant
   eps by default), but both T7 arms were launched with
   `'mcts.root_exploration_eps_schedule=[[0,0.25],[3000000,0.05]]'` passed
   explicitly — the grid matching its temperature schedule. The deciding
   evidence was the run-through benchmark: without the anneal T7 would produce
   another set of checkpoints whose headline rate does not survive greedy eval,
   which is exactly what 7 of 12 T6 specialists turned out to be.
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

## T3 — Capacity / horizon (**not needed so far** — T2 closed enough of the gap that T6/T7 ran on the existing net)

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

## T4 — The curriculum run (**done** — benchmark leg passed 2026-08-12, the real run launched as T7 on 2026-08-24)

`scripts/submit_curriculum.sh` was written 2026-07-20 and committed in `ed4808d`,
and was finally submitted 2026-08-24 — not as `curriculum-v1` but as T7's two
named arms, since the T7 comparison is the reason the run exists. 12 levels,
60M env steps, 4 chained 24h legs, 2 GPUs, 64 workers, 1M buffer.

The benchmark leg **passed** as job **5989959** (2026-08-12): 200K env steps
in 30 min, ~111 env-steps/s on the 2-GPU split, which is what cleared T7's
~384 GPU-h commitment. It also now exercises the T2.3 env, since SLURM reads the working
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

## T6 — The specialist fleet (**done 2026-08-16**; 1-3/4-3 rescued 2026-08-20)

12 single-level runs, 15M env steps each, matching the diag-run budget so the
results are comparable to `level1-1-diag-v1` (0.84) and `level1-2-diag-v1`
(0.68). Run dirs `outputs/runs/spec-<level>/`.

**Fleet #1** (jobs 5990729-5990752, launched 2026-08-11) ran 2h15 and was
killed fleet-wide by the hard 101 GB home quota — 12 × 271 MB checkpoints
every 16 min. Every leg-1 died at its next save boundary; every leg-2 was
cancelled on the dependency. The ~40K-train-step checkpoints survived, and
the 2h15 of production self-play did validate both new mechanisms (eps anneal,
stochastic starts) and showed spec-level3-2 at 0.14 completion by step 42K —
ahead of the diag pace. **Fleet #2** (jobs **6017303-6017326**, launched
2026-08-14) resumed those checkpoints via the normal `latest.pt` path, with
storage now on `/projects` and failed saves made non-fatal, and **finished
2026-08-16** — all 12 reached `TRAINING_COMPLETE` at 15M env steps with no
quota deaths. Per-level rates are in the table at the top; 1-3 and 4-3 came
out at 0.00 and were handled separately (see the imitation rescue below).

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

## T7 per-level read (**done 2026-09-07, job 6382949**)

The 0.11-vs-0.02 headline is a pooled self-play rate. Per level, greedily, at
`latest.pt` for both arms (matched at 60.0M env steps):

**Both arms complete 0 of 12 levels.** Not one greedy completion between them.
Whatever the human arm's higher self-play rate is measuring, it does not survive
the removal of exploration noise — the same gap the specialists show, and the
reason the fleet's headline rates were never the deployed performance.

By **return**, which does separate them, the human arm leads on **8 of 12**:

| | human | nohuman | |
|---|---|---|---|
| 1-1 | **243.5** | 65.1 | human |
| 1-2 | **46.1** | 12.9 | human |
| 1-3 | **80.0** | 48.7 | human |
| 2-1 | **82.1** | 37.1 | human |
| **2-2** | 84.9 | **126.3** | **nohuman — the control** |
| 2-3 | -19.3 | -20.3 | tie, both stuck |
| 3-1 | 29.5 | **62.0** | nohuman |
| 3-2 | **200.0** | 28.1 | human |
| 3-3 | **65.7** | 38.0 | human |
| 4-1 | 33.4 | **38.9** | nohuman |
| 4-2 | 26.3 | 26.5 | tie |
| 4-3 | **67.2** | 46.3 | human |

**Level2-2 is the control and it behaves as a control should.** It is the one
level with zero human data, and it is one of the levels where the no-human arm
wins — the human arm has no advantage exactly where it had no demonstrations to
learn from. That is the pattern you would want if the advantage is really coming
from the demos rather than from a lucky seed.

**Re-run at n=5 with stochastic starts (jobs 6383784/6383785) — the negative
result holds and hardens.** Across 60 rollouts each: `curriculum-human`
completes **1** (Level1-1, 694 steps), `curriculum-nohuman` completes **0**.
1/60 against 0/60 is not a difference worth defending, so the honest statement
is that **neither curriculum arm can finish any level greedily**, and T7's
0.11-vs-0.02 lives entirely in noisy self-play. The n=1 return-based split
(8 of 12 to the human arm, with Level2-2 the control) is not contradicted by
this, but it was one trajectory per cell and remains unconfirmed — returns were
not re-collected at n=5.

## T10 — Lesion study (**machinery built + pilot run 2026-09-07**)

Port of the Towers-of-Hanoi region-specific-planning experiment
(`~/region-specific-planning/Muzero-Hanoi`) to Mario: re-initialise a targeted
component of a trained net at evaluation time and measure the deficit.
`src/muzero/lesion.py` keeps that study's three conventions unchanged so the two
are comparable — lesion = random re-initialisation (not noise, not zeroing),
evaluation-time only, and lesioning one target leaves every other parameter
bit-identical (`tests/test_lesion.py`, 11 tests).

**Mario needed a distinction Hanoi did not.** Hanoi's policy/value/reward heads
are standalone Linear stacks off the latent; Mario puts policy and value behind
a **shared residual trunk** (`prediction.blocks`) and the reward head inside
`DynamicsNet`. So `policy` resets only the policy-specific conv/bn/fc — a
careless `prediction`-wide reset would destroy value too and report a policy
deficit that is really policy+value. Mario also gains two targets Hanoi has no
analogue for: **`transition`** (the latent forward model MCTS rolls out) and
**`encoder`**.

**Pilot (job 6380549, Level3-3 `spec-level3-3` best.pt, 2 rollouts x 2 lesion
seeds, greedy):**

| Condition | completes | x median | params reset |
|---|---|---|---|
| intact | 1.0 | 2498 | 0 |
| value | **1.0** | **2498** | 61,324 |
| reward | **1.0** | **2498** | 87,425 |
| value+reward | 0.5 | 2278 | 148,749 |
| transition | 0.0 | 1063 | 6,996,096 |
| policy+value | 0.0 | 440 | 62,590 |
| pred_trunk | 0.0 | 419 | 664,320 |
| policy+reward | 0.0 | 374 | 88,691 |
| **policy** | **0.0** | **357** | **1,266** |
| policy+value+reward | 0.0 | 256 | 150,015 |
| encoder | 0.0 | 166 | 4,488,384 |

**Three things worth following up, and the first is the headline:**

1. **The deficit is not proportional to damage — it is almost inverse.**
   Destroying the policy head's **1,266** parameters (0.006% of 22.6M) takes the
   model from finishing the level to dying at x=357. Destroying the 7.0M-parameter
   forward model still leaves it reaching x=1063, three times further. Whatever
   this agent is doing, the policy prior is doing it.
2. **Value and reward lesions are individually free** — 1.0 completion, identical
   x, step counts within noise of intact (382-393 vs 387-391). That is the
   *opposite* of the Hanoi phenotype, where the value lesion is the one that
   produces the PFC-like deficit. The obvious hypothesis is that MCTS here is
   prior-dominated: at 50 simulations over 12 actions the tree is shallow, so the
   value head barely enters action selection. **The control that would settle it
   is a `mcts.num_simulations` sweep** — if value only becomes load-bearing at
   higher simulation counts, that is a statement about how much planning this
   agent does, and it connects directly to the existing finding that these models
   stall at fixed obstacles and lean on exploration noise.
3. **`value+reward` together (0.5) is worse than either alone (1.0, 1.0)** —
   a superadditive interaction worth confirming with more seeds.

### Replication across 3 levels overturns the pilot's "value is free" (job 6380882)

n=12 per lesioned condition, n=3 intact, 50 sims.

| level | teacher | intact | value | reward | policy | transition |
|---|---|---|---|---|---|---|
| Level3-3 | self-play | 1.00 | **0.917** | 0.917 | 0.00 | 0.167 |
| Level6-1 | self-play | 1.00 | **0.417** | 0.667 | 0.00 | 0.083 |
| Level1-3 | **imitation** | 1.00 | **0.083** | 0.833 | 0.083 | 0.083 |

**The pilot's headline was a single-level artefact.** "Value and reward lesions
are individually free" is true on Level3-3 and false elsewhere: value falls to
0.417 on Level6-1 and to **0.083** on Level1-3, where it is as damaging as the
policy lesion (0.083) and drops `final_x` 2514 -> 1266. Reward is mild but not
free either (0.667-0.917, where the pilot at n=4-6 read 1.00).

**What survives:** the policy lesion is catastrophic on all three levels, and it
is still only **1,266** parameters. Transition is catastrophic on all three
(0.083-0.167) despite the model retaining an intact encoder and both heads.
Encoder is 0.00 everywhere.

**The confound to resolve before writing any of this up.** Level1-3 — the level
where value matters most — is the **only imitation-trained model in the set**
(`spec-imit-level1-3`); Level3-3 and Level6-1 are pure self-play. So
"value-lesion severity varies by level" has a rival explanation, "varies by
training recipe": a BC-pretrained model may lean on its value head differently.
Jobs **6384697** (Level8-2, Level5-2 — imitation) and **6384698** (Level1-1,
Level3-2 — self-play) add two models of each kind to separate these. Note the
comparison is not competence-matched — no two models of different recipe have
the same self-play rate — so read it as a direction, not a clean contrast.

### The recipe-vs-level confound test does not work with the models we have

Job **6384697** (Level8-2, Level5-2 — both imitation-trained) came back
**vacuous, and the reason generalises**: a lesion deficit is measured against an
intact baseline, and these models have no baseline to speak of.

| level | intact (n=3) | value | policy | reward |
|---|---|---|---|---|
| Level5-2 | **0.333** | 0.0 | 0.0 | 0.0 |
| Level8-2 | **0.000** | 0.0 | 0.0 | 0.5 (!) |

Level8-2's *intact* model completes 0 of 3 greedy rollouts, so every condition
reads 0.0 and the lesion is unmeasurable; its reward row at 0.5 is noise against
an n=3 baseline of zero, not a lesion that helps.

**The constraint this exposes: the lesion study needs models that complete
reliably when intact**, and the 2026-09-05 benchmark says only a handful do —
1-3, 3-3 and 6-1 at 5/5, then 3-2 and 6-3 at 4/5, 5-2 at 3/5. Of those, exactly
**one is imitation-trained (Level1-3)**, which is the very model the confound is
about. So **recipe and level cannot be separated with the current model set**;
it is not a matter of running more rollouts. Resolving it needs either a
strong imitation-trained model on a level that also has a strong self-play
one, or the lesion grid re-run under noisy (non-greedy) evaluation where weak
models still show gradations. Left open deliberately.

Also note **intact n=3 is too small for any of this** — it is the denominator
every deficit is read against. Job 6382950 fixed that for the Level3-3 sweep
(n=8); nothing else has.

### The simulation sweep answered #2, in the opposite direction to the guess

Job **6380994** (Level3-3, 2 rollouts x 3 lesion seeds per cell). The guess was
"value is free because search is shallow; deepen it and value will start to
matter". The direction is right and the mechanism is the reverse of benign:

| sims | intact | value | reward | transition x | policy x |
|---|---|---|---|---|---|
| 10 | 1.00 | **1.00** | 1.00 | 1463 | 375 |
| 25 | 0.50* | **1.00** | 1.00 | 893 | 407 |
| 50 | 1.00 | **0.83** | 1.00 | 884 | 356 |
| 100 | 1.00 | **0.67** | 1.00 | 776 | 203 |
| 200 | 1.00 | **0.50** (x 1641) | 1.00 | 636 | 208 |

\* n=2 per intact cell; the 25-sim 0.50 is noise, not a dip.

**Deeper search amplifies the damage from a broken value head.** Intact is
**flat** across simulation counts — extra search buys the healthy model nothing
— but the value-lesioned model falls from 1.00 to 0.50 as simulations go 10 ->
200, and its `final_x` finally breaks off the ceiling at 200 (2498 -> 1641).

**Correction (job 6382950, n=8 per intact cell, 2026-09-07):** the sweep table's
intact row was n=2 and read as a clean 1.00 at every count. At n=8 it is
**0.875 / 0.875 / 1.00 / 0.875 / 1.00** for 10 / 25 / 50 / 100 / 200 — flat, with
no trend, but ~0.9 rather than a hard ceiling. The comparison survives intact
(flat baseline against a monotonically declining value-lesion row) but "sits at
ceiling" was an artefact of n=2 and should not be repeated.
With a randomly re-initialised value head, more planning is actively worse than
less: the corrupted bootstrap gets propagated into the root by exactly the
mechanism that is supposed to make search helpful. The same monotone
degradation shows in `transition` (1463 -> 636), which is the other component
the tree consults on every simulation.

**`reward` is free at every depth** (1.00 across the sweep) — the one component
this agent genuinely does not use. Worth a thought: Mario's shaped reward is
dense, so the value head may already carry everything the reward head would say.

**Read together with the pilot, the Level3-3 phenotype is:** policy =
catastrophic and search-depth-independent; value = latent, and only expressed
under deep search; reward = silent; transition = catastrophic and
depth-amplified. **But the 3-level replication above shows this phenotype is
Level3-3's, not the agent's** — value is far from silent on Level6-1 and
Level1-3. The depth-dependence of the value lesion has only been measured on
Level3-3; job **6383757** repeats the sweep on Level6-1, where value already
costs 0.58 at the default 50 sims, so the depth effect there should be easier
to see, not harder. That is a
different phenotype from Hanoi's, where the value lesion is the headline deficit
at the default search budget — and the difference is now *measured* rather than
assumed.

**Caveats:** one level, one checkpoint, n=6 per lesioned cell and n=2 per intact
cell. The intact row needs more rollouts before the sweep is publishable; it is
the baseline every other row is read against.

**In flight at the pause (launched 2026-09-07):**
- **6380882** — T10 replication of the 50-sim grid across Level3-3 / Level6-1 /
  Level1-3, 3 rollouts x 4 lesion seeds (369 rollouts) →
  `outputs/lesion/lesion_eval-6380882.json`.
- **6382950** — T10 intact baseline, 8 rollouts at each of 10/25/50/100/200
  simulations. This is next-action #6: the sweep's intact row was n=2 per cell
  and is what every other row is read against.
- **6382949** — T7 per-level read (next-action #5), both arms at `latest.pt`
  (matched at 60.0M env steps) across all 12 curriculum levels with
  `--compare-human` → `outputs/t7_per_level/<arm>/`. **Level2-2 is the control**:
  zero human data, so if arm B wins there too the win is not from the demos.
- **6382967-6382990** — T8 step 1 (next-action #4), the specialist teacher
  corpus, **fanned out one job per level** across all 23 →
  `outputs/specialist_trajectories/`, one `dump_report_<Level>.json` each.
  Fanned out because cost is dominated by weak specialists: a level completing
  at 0.03 burns its whole attempt budget for a handful of episodes, and 23 of
  those in series would blow the wall. `--max-attempts-per-level 60` bounds it.
  **These reports are the input to the teacher-quality-bar decision** — they say
  which levels came up short, which is exactly what that decision needs.

## Comparison bundle — old models vs new (**built 2026-09-05**)

Supervisor asked to compare earlier checkpoints against the current fleet, on
the observation that the newer ones align better with certain brain regions.
**Checked whether the intervening code changes invalidate that comparison: they
do not.** From the oldest run's sha (3bac0a8, July) to HEAD, `networks.py`,
`transforms.py` and `preprocess.py` are **byte-identical**; the 1,118 changed
lines under `src/` are all learner / buffer / MCTS / worker / human-comparison
machinery, and `env.py`'s 87 are purely the stochastic-start logic (the
frames→observation mapping is untouched). All candidate runs also share the
identical architecture (192ch, [2,2,2,2], 10 dyn, 22.6M params), observation
pipeline (4 x skip-4 @ 96) and value/reward supports ([-25,25,201]). So the
same frames through an old and a new checkpoint give directly comparable
(N,192,6,6) activations — verified by running one frame batch through four of
them.

**The confound is not the code, it is that the four old runs are not
equivalent to each other:**
- `level1-1-diag-v1` (0.84) / `level1-2-diag-v1` (0.68) ran the *same* discount
  0.999, bonus 200 and 15M budget as the current specialists, differing only in
  stochastic starts and the eps anneal — the cleanest old-vs-new pair available.
- `imit-all-v1` / `imit-sub01-v1` ran **discount 0.997, completion_bonus 100**,
  i.e. pre-discount-fix, on 10M steps, 12 levels pooled, imitation on. Against
  the current specialists that changes seven things at once plus raw competence,
  so an alignment difference there is uninterpretable. **Compare them against
  each other** (identical but for all-subjects vs sub-01 demos), where the
  confound cancels.

`exports/muzero_mario_comparison_20260905.zip`, 8 checkpoints, 396 MB, with
three intended pairs documented in its generated README. Curriculum arms ship
`latest.pt`, not `best.pt`, deliberately: arm A's best is from env step 23.9M
where it plateaued, so best-vs-best would have compared 2.29M against 1.13M
train steps and silently broken the matched pair.

`package_models.py` gained what this needed: `--include-run RUN[:CKPT]` (ship a
named run keyed by run name rather than level, so old and new models of the same
level do not collide on `<level>.pt`, with an optional forced checkpoint),
`--note` (a bundle-specific paragraph in its README) and a generated
contents-table header so a bundle whose composition differs from the per-level
fleet describes itself accurately.

## T7 — Autocurriculum ± human teacher (**DONE 2026-09-05 — the human arm wins**)

**Both arms reached the full 60M env-step budget** (jobs 6249958-6249963; leg 3
of each ended `COMPLETED`, not `TIMEOUT`, so the 3 extra legs were right-sized).
At equal budget:

| Arm | Env steps | Train steps | Best pooled rate |
|---|---|---|---|
| `curriculum-human` | 60.0M | 2.80M | **0.11** |
| `curriculum-nohuman` | 60.0M | 2.70M | 0.02 |

**5.5x, and arm A never moved off 0.02 after 23.9M env steps** — a 36M-step
plateau, not noise. The human teacher is what let the 12-level curriculum learn
at all. Read it as a directional result, not a per-level one: it is a pooled
rate over 12 levels, and Level2-2 (the zero-human-data control) still needs
checking per level before the win is attributed to the demos.


**T7 ran out of legs and nobody noticed for four days.** Both arms were
submitted with `submit_curriculum.sh`'s default `N_LEGS=4` against a 60M
env-step budget. Every leg exited `TIMEOUT 0:0` at the 24h wall — which *is*
the designed resume path — and the fourth leg simply had nothing to hand off
to. Final leg ended 2026-08-28T19:56 (arm A) / 19:58 (arm B) at
**39.4M and 36.0M env steps**, no `TRAINING_COMPLETE`, nothing in the queue,
`.err` files containing only wandb banners. A run out of legs is
indistinguishable from a finished one at a glance.

The arithmetic was wrong at launch, and was knowable then: `bench-2gpu`
measured ~111 env-steps/s on 2026-08-12. Arm A managed 114/s (39.4M in 96h),
arm B 104/s (36.0M — slower for the 50K BC pretrain, which collects no env
steps, plus mixed-batch cost). 60M therefore needs ~6.1 / ~6.7 legs, not 4.

**Fix, 2026-09-02:** 3 more legs each, jobs **6249958-6249960** (arm A) and
**6249961-6249963** (arm B), resumed from `latest.pt` with the arms' recorded
overrides re-passed verbatim from their `.hydra/overrides.yaml`. Arm B keeps
`imitation.enabled=true` even though its mix ratio annealed to 0 at train step
750K and contributes nothing further: `human_compare` scores action-agreement
on the imitation loader's holdout *only when imitation is enabled*, so turning
it off mid-run would silently start scoring on files the model trained on.
Resuming past `pretrain_steps` is safe (start_step 1.75M >> 50K, pretrain is a
no-op).

**Guard added so this cannot recur silently:** `submit_chain.sh` now estimates
whether `N_JOBS` can reach `training.total_env_steps` at ~110 env-steps/s
(`MUZERO_ENV_STEPS_PER_SEC` overrides) against the leg script's own
`#SBATCH --time`, and warns with the leg count actually needed. It subtracts
progress already banked in `latest.pt`, so a resumed chain is not told to
re-budget the whole run. `MUZERO_CHAIN_DRY_RUN=1` runs the check and submits
nothing. Validated against the failure it exists for: it predicts 4 legs reach
~38.0M against the 39.4M/36.0M actually observed.

**Early read (weak — a 12-level pooled rate):** arm B 0.08 at 36.0M, arm A
0.02 at 39.4M and flat since 23.9M. The human arm is ahead at fewer env steps.



Two runs off `submit_curriculum.sh`, identical except for the `imitation:`
block:

Launched 2026-08-24 as jobs **6115540-6115543** (arm A) and **6115544-6115548**
(arm B), exactly these commands:

```bash
bash scripts/submit_curriculum.sh curriculum-nohuman 4 \
    'mcts.root_exploration_eps_schedule=[[0,0.25],[3000000,0.05]]'
bash scripts/submit_curriculum.sh curriculum-human 4 \
    imitation.enabled=true \
    imitation.pretrain_steps=50_000 \
    'imitation.mix_ratio_schedule=[[0,0.25],[700_000,0.0]]' \
    'mcts.root_exploration_eps_schedule=[[0,0.25],[3000000,0.05]]'
```

The eps anneal is on **both** arms so it stays a controlled variable; the
completion gate is left off, since it keys on the run's first completion of
*any* level and an easy level unlocks it almost immediately in a 12-level run.

Chosen 2026-07-29: **pretrain + annealed mix**, all subjects. The anneal is the
point — a one-hot BC anchor that never fades drags on the policy indefinitely,
which is what `mix_ratio_schedule` was added for. Schedule thresholds are RL
train steps and the 50K pretrain offset is applied automatically.

**The benchmark leg was run first** and passed (job 5989959, 2026-08-12):
~111 env-steps/s on the 2-GPU split.

**Caveat — the human arm covers 11 of 12 levels.** Verified against the corpus
2026-07-29: there is no `level-w2l2`, i.e. **Level2-2 has zero human data**
(humans also never played w7l2 or the castle -4 levels, but those are not in
`env.levels`). The autocurriculum will keep sampling Level2-2 and the human
buffer contributes nothing there, so any T7 win must be checked per level
before it is attributed to the human teacher. `levels: match_env` filters by
filename tag and will simply find no w2l2 files. **Confirmed safe 2026-08-24**:
`select_human_files` filters on "matches any requested tag", so the 12-level
request returns 3,926 files across the other 11 levels and never raises;
Level2-2 contributes 0. `human_compare`'s per-level path skips levels with no
human data (tested). Still to confirm on arm B's first leg: that the ~54 GB
corpus load fits the 220 G/job (110 G x 2 GPUs) alongside the 1M-transition
buffer.

**Level2-2 is the natural control for this arm.** It is the one level where the
human teacher contributes nothing, so if arm B beats arm A *there* too, the win
is not coming from the demos.

## T9 — Brain-data level coverage (**self-play pass done 2026-09-03**)

**All 11 reached `TRAINING_COMPLETE` at 15M env steps** (jobs 6238199-6238220),
finishing 2026-09-03. Seven completed their level at least once:

| Level | Best rate | First completion (RL step) |
|---|---|---|
| 6-1 | **0.93** | 39,848 |
| 5-1 | **0.73** | 102,126 |
| 6-3 | **0.70** | 181,493 |
| 7-1 | 0.48 | 30,883 |
| 6-2 | 0.25 | 125,070 |
| 8-3 | 0.03 | 150,632 |
| 8-1 | 0.01 | 619,389 (82% of budget) |
| 5-2, 5-3, 7-3, 8-2 | **0.00** | never |

Level6-1 at 0.93 ranks third across the whole fleet behind 3-2 and 3-3, so
worlds 5-8 are not uniformly out of reach. But 4 of 11 dead against T6's 2 of
12 confirms they are harder on average, and 8-1's 0.01 at 82% of budget is the
same cracked-open-not-solved shape as 4-3.

**Rescue outcome (done 2026-09-05, jobs 6274783-6274790): 3 of 4 rescued, and
the prediction below was backwards.**

| Level | Human completions | Result | First completion |
|---|---|---|---|
| 8-2 | **4** | **0.52** | RL step 164k |
| 5-2 | 13 | 0.25 | 339k |
| 7-3 | 15 | 0.19 | 310k |
| 5-3 | **19** | **0.00** | never |

The level with the *fewest* human completions rescued best and the one with the
most never completed once. **Human demo count does not predict rescue success** —
whatever governs it is the failure geometry, not teacher volume. Do not reuse
the heuristic below for triage.

**Level5-3 is the one dead level**: 15M env steps of pure self-play and 15M with
imitation, zero completions in either. Greedy rollouts (job 6338191) show both
checkpoints dying — never timing out — at a fixed wall: `spec-level5-3` at
x=547-907 (2 of 5 die early at ~550), `spec-imit-level5-3` at x=906-910 on all
5. Same obstacle, the imitation model just reaches it reliably.
**Shipped anyway, as `spec-level5-3/latest.pt`** (no `best.pt` exists — it is
only written on a completion-rate high): the ~3% distance the imitation model
gains does not pay for a BC-pretrain confound on the exact level a collaborator
would analyse against w5l3 brain data.

Original launch note, kept because the prediction was wrong and that is the
useful part:
**Rescue pass launched 2026-09-03** for the four zeros — jobs 6274783-6274790,
`submit_specialist_imitation.sh` on Level5-2/5-3/7-3/8-2. **Expect less than
the 1-3 rescue delivered**: that level had 78 human completions to teach from
and escaped to 0.85, while 4-3 had 60 and reached only 0.01. These have 13
(w5l2), 19 (w5l3), 15 (w7l3) and **4** (w8l2). Thin teachers are the whole
risk here.

**Final packaging (2026-09-05):** greedy benchmark re-run as job 6338095 across
all 22 levels with a `best.pt` (the 2026-09-03 run predates the rescue and
misses 5-2/7-3/8-2), with `submit_package_models.sh` queued behind it on
`afterok` as job 6338250. `package_models.py` gained `--include-incomplete`
(fall back to `latest.pt` for a level that never completed, marked
`completed_level: false` in the manifest so it is never passed off as a best)
and `--pin LEVEL=RUN` (force a run, used to take the self-play 5-3 over the
imitation one on measured evidence rather than alphabetical order). Bundle goes
to **23 levels, ~1.1 GB zipped**, covering 22 of the 22 brain-data levels plus
Level2-2.

**Earlier greedy benchmark 2026-09-03**, job 6274769, covering
all 23 levels with a trained run (the 4 with no `best.pt` skip themselves).
`eval_human_benchmark.py`'s hard-coded 12-level `ALL_LEVELS` now derives from
the run dirs, as `package_models.py` already does.



Trigger (2026-09-01): a collaborator using these checkpoints against CNeuroMod brain data
asked for models on every level they have brain data for. The bundle shipped
2026-08-28 covers 12 levels; the human corpus covers 22. The overlap is 11 —
**Level2-2 has a model but no human data, and w5l1-w8l3 have human data but no
model.** So half the levels with brain data had nothing to compare against.

Why the gap existed: the 12-level `env.levels` list predates the git history.
`conf/env/mario.yaml` is absent from the initial commit (the code read
`cfg.env.levels` against an untracked file) and first appears already formed in
f91adf7. Most likely inherited from the `ppo_study` PPO baseline this repo is
deliberately a sibling of — unverifiable here, since that repo was on BMRC and
is not on Isambard. Either way it was chosen to serve a PPO comparison, not the
brain-data one, and nobody revisited it when the human corpus landed.

**Scope: 11 levels** — Level5-1/2/3, Level6-1/2/3, Level7-1, Level7-3,
Level8-1/2/3. Not w7l2 (humans never played it, so no brain data) and not the
castle -4s (same, plus a different terminal condition).

**Recipe: pure self-play** (`submit_specialist.sh`, the exact T6 recipe), chosen
deliberately over the imitation variant even though human demos with completions
exist for all 11 (277 completions; thinnest are w8l2/w8l3 at 4 each). Reason:
a model BC-pretrained on the same subjects' gameplay the fMRI comes from is
confounded as evidence that its representations resemble those subjects' brains.
The current bundle already mixes both kinds — 1-3 and 4-3 are imitation-trained,
the other 10 are not — which is worth flagging to anyone analysing them as a set.
Expect failures: worlds 5-8 are harder than 1-4 and the T6 recipe already
produced 0.00 on 2 of 12 easier levels. Failures get the imitation rescue as a
second pass, exactly as 1-3/4-3 did.

Both submit scripts' hard-coded 12-level `VALID` gate now derives from the
state files under `mario.stimuli/SuperMarioBros-Nes/`, so worlds 5-8 need no
further edits. All 11 levels were verified to load, step and accrue shaped
reward before launch (env-only check, login node).

## T8 — Distillation into one all-level model (**unblocked; dumper written 2026-08-24**)

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

**Step 1 — the dumper (`scripts/dump_specialist_trajectories.py`, WRITTEN
2026-08-24).** Loads each specialist checkpoint, rolls out its own level, and
emits `Trajectory` .npz in the same format as `convert_human_bk2.py`.
Round-trip verified end to end against the untouched loader (real Level3-3
episodes: glob, level filter, subject filter, `load_human_buffer`, and a
sampled training batch). **Two corrections to the plan as originally written:**

- **Filenames must start with `sub-`.** `select_human_files` globs
  `sub-*.npz`, so the `spec-w1l1_...` naming proposed here would have been
  silently invisible — no error, an empty corpus. Files are written as
  `sub-spec-w1l1_ses-000_task-mario_level-w1l1_rep-000_seg0.npz`, which keeps
  the zero-loader-change property and still makes the teacher its own
  selectable subject (`sub-spec-w1l1`). Guarded by
  `tests/test_specialist_dump.py`.
- **Rollouts need exploration noise.** The benchmark run showed 7 of 12
  specialists cannot complete their own level greedily from `best.pt`, so a
  greedy dumper would give those levels a corpus with zero successful
  demonstrations. Rollouts default to `--eps 0.25` / `--temperature 0.25` and
  keep only completing episodes; the script reports levels that fell short of
  the target so a thin teacher cannot pass unnoticed.

This also resolves the standing "rethink `best.pt` selection before T8" item,
but not the way it was framed: `best.pt` is picked on the *noisy* rolling rate,
which is a poor predictor of greedy competence — yet for **corpus generation**
under noise it is exactly the right selection criterion. The concern applies to
deploying a specialist, not to using it as a teacher.

Store the **raw MCTS visit distribution** in `policies`, not a one-hot (done —
verified 0/404 one-hot rows on real dumps, mean max-share 0.26), so the policy
loss gets genuine distillation targets rather than behavioural cloning.
`root_values` are real MCTS root Q, so returns and priorities are built exactly
as `src/selfplay/worker.py` does.

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

- **2026-08-24** — **T7 launched**: `curriculum-nohuman` (6115540-6115543) and
  `curriculum-human` (6115544-6115548), 4 legs each, both with the eps anneal
  `[[0,0.25],[3000000,0.05]]`. Also: T8 dumper written and round-trip verified
  (`58eeb56`), backlog brought current, 5.8 GB stray `core` removed from the
  repo root.
- **2026-08-24** — **Run-through benchmark** (job 6083435) →
  `images/human_vs_agent_runthrough.pdf`. Each level's `best.pt` played through
  its level 5x with stochastic starts: **5/12 levels finish at all, and every
  one that finishes beats the human median time**; 7/12 never finish. 1-1
  stalls at exactly x=2370 from five different starts despite a 0.72-0.91
  self-play rate. This is the clearest measurement yet of how much the fleet's
  headline numbers depend on exploration noise.
- **2026-08-21** — **Human comparison shipped** (`ab0ceab`): `human_compare:`
  scores every checkpoint replay against a human on the same level. Fixed two
  scale bugs found doing it — the comparator must be the human *first-life*
  rate, and `run_replay_rollout` had never been passed the run's
  `completion_bonus`, so `replay/<level>_return` was mis-scaled in every run
  to date.
- **2026-08-20** — **Imitation rescue finished** (`0c9aed7`): `spec-imit-level1-3`
  escaped outright (0.00 → **0.85**, greedy replay completes, faster to the flag
  than the median human); `spec-imit-level4-3` only cracked open (0.01, first
  completion at 79% of budget).
- **2026-08-16** — **T6 fleet #2 finished**: all 12 specialists reached
  `TRAINING_COMPLETE` at 15M env steps. Level1-3 and Level4-3 ended at 0.00
  completions, which is what motivated the imitation rescue.
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
