# Stall-triggered sampling experiment, v1

**Completed:** validation passed 21 tests and all smoke checks; all 270 full
development episodes completed without infrastructure errors. See
[STALL_SAMPLING_RESULTS_V1.md](STALL_SAMPLING_RESULTS_V1.md) for the results and
trace audit. Queue-status notes below preserve the initial submission record.

The user authorized testing greedy control with brief seeded exploration after
a stall. This is an external evaluation controller on frozen checkpoints; no
network, training objective, BatchNorm setting, or production MCTS change.

## Frozen controller

Before each decision, read the current game x-position and player-state flag.
While player_state=8 (normal control), retain the last **97 positions**, spanning
**96 decisions**. If their maximum minus minimum is **at most 16 pixels**, start
a fixed **32-decision** burst of MCTS action sampling with temperature **.25**.
After that burst, return to temperature 0 and clear the window. Another trigger
requires 96 fresh greedy decisions. Non-control states clear history and cancel
sampling. There is no position-specific rule, obstacle detector, forced jump,
level-specific setting, root noise, or search-budget change.

At nominal four-frame decisions these intervals correspond to about 6.4 seconds
of low movement and 2.1 seconds of sampling at 60 Hz. The 96/16 trigger follows
the already tested stall criterion; 32 decisions is one predeclared initial
burst length intended to allow several action changes. It has not been selected
by a sweep. Small-progress stalls and sustained wider oscillations may evade
this criterion; useful pauses may trigger it. Both are limitations to measure.

The monitor uses simulator **RAM-derived x and control state**, not solely model
visual activations. The neural network still receives its original image input.
Any deployed/fMRI system using this controller must describe this extra control
logic; its behavior is not a pure visual policy-head output.

## Comparison and samples

Levels: **1-1** (candidate), **6-1** (required successful control), **1-3**
(additional strong control vulnerable to always-on sampling). Freeze the original
v1b best.pt for 1-1/6-1 and coverage-v1's selected imitation best.pt for 1-3,
with expected SHA256 checks. Three arms use identical checkpoints and starts:

| Arm | Action selection |
|---|---|
| greedy | Temperature 0 throughout |
| sampled | Temperature .25 throughout |
| stall_sampled | Temperature 0 except during triggered .25 bursts |

All arms use leaf batch 4, 50 simulations, root epsilon 0, checkpoint reset
settings, and a 2000-decision cap. Use the original **30 development seed pairs**
on each level, interleaved across arms: **270 episodes** total. This reuses
development conditions for diagnosis; it is not independent confirmation.
Freshly rerun baselines verify the controller's effect on identical checkpoints.
The monitor consumes no randomness, so gated and greedy action/search streams
are identical until the first trigger, barring numerical nondeterminism.

Reserve **100 new confirmation pairs**, environment seeds 5100001–5100100 and
search seeds 6100001–6100100. They are disjoint from development and the consumed
3100001/4100001 confirmation ranges. **Do not run these yet.** Review development
behavior, freeze any selected rule, and then use an untouched confirmation set.
If the rule is changed after testing these new seeds, another fresh set is needed.

## Measurements and execution

Primary: all-attempt completions, deaths and timeouts, with paired rescues/losses
and n=30 explicitly stated. Also record every trigger's decision index and x,
number of bursts, sampled fraction, per-decision temperature and sampling state,
and the existing full action/visit/prior traces. Distinguish interventions on
episodes that greedy would complete from rescues of greedy failures. Do not
describe lower sampling time alone as better performance.

Validation runs on a Slurm GPU: targeted state-machine/accounting tests, one
episode per arm on all three levels, and exact comparison of both 6-1 baseline
traces against the original development experiment. Smoke trials do not add to
the full denominator. Full jobs run per level after validation succeeds; frozen
source, checkpoints and traces live under `/projects` at
`outputs/controller_policy/stall-sampling-v1-20260911/`.

## Submitted jobs

| Task | Job |
|---|---:|
| Preparation, targeted tests, smoke checks | 6464069 |
| Level1-1, 30 trials × 3 controllers | 6464076 |
| Level6-1 control, 30 trials × 3 controllers | 6464077 |
| Level1-3 control, 30 trials × 3 controllers | 6464078 |

At submission audit, validation was pending GPU resources and the three full
jobs were pending its afterok dependency, with kill-on-invalid-dep enabled.
No tests or smoke results have yet been reported for this experiment. Read
Slurm status and output logs before claiming validation succeeded or interpreting
full results. The new 100-pair confirmation list is reserved only, not submitted.
