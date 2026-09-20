# Local stall interventions, 2026-09-11

**Completed:** full jobs 6459327/6459328 succeeded. See
[CONTROLLER_INVESTIGATION_RESULTS_20260911.md](CONTROLLER_INVESTIGATION_RESULTS_20260911.md)
for all 19-root outcomes and interpretation; submission status below is historical.

Purpose: test whether short action sequences are sufficient to release the
Level1-1 low-progress state, while measuring their effects on Level6-1. This
addresses the **stall signature**, not the death-dominated Level8-1 failure.
It changes evaluation only. No training, weights, or production search change.

## Frozen selection and intervention rules

Use the original completed v1b development greedy traces only. Among all 30
Level1-1 trials, consider every timeout (19 in the completed experiment). For each,
choose the first bank observation with x>256 whose preceding 96 decisions plus
current observation span at most 16 x pixels. Require at least 128 decisions
remaining in the original trace. Record cases that fail this eligibility rule;
do not change the rule to obtain a desired count. There is at most one root per
source episode. This is a conditional sample of stalls, not all gameplay.

For each retained episode index, use the same Level6-1 greedy episode, without
filtering its outcome. Choose its saved observation nearest the same fractional
elapsed episode time, retaining eight available source decisions. This is a
control for the intervention's general effects, not matched terrain across
levels. Report the control cohort's original intact completion count.

Five eight-decision prefixes (nominally 32 emulator frames):

| Branch | Prefix |
|---|---|
| logged_prefix, reference | The next eight original greedy actions |
| release | NOOP for eight decisions |
| right_jump | Right+jump for eight decisions |
| right_run_jump | Right+run+jump for eight decisions |
| left_then_right_run_jump | Left for four, then right+run+jump for four |

After the prefix, return to the unchanged greedy MCTS controller for the rest
of a **128-decision total horizon**, or until termination. Use batch 4, 50
simulations, root noise 0, temperature 0. Continuation search seed is fixed at
9100001+episode_index and shared across branches. No search runs during any
prefix, including the reference prefix, so its RNG consumption is matched.
The reference is newly seeded continuation after eight recorded actions;
it need not duplicate the original full greedy timeout. Measure its behavior
rather than assuming it remains stuck. No confirmation seeds/data are used.

## Exact branch reconstruction

Create a fresh emulator/wrapper for every branch and replay the recorded
actions from the original environment seed. Verify reset provenance, x/reward
at every replayed decision, and exact uint8 equality to the saved root observation.
This avoids relying on an emulator-only snapshot that omits frame stacks, reward
counters, or scenario state. Refuse divergent replays rather than analyzing
mismatched roots. Save root and branch-end screenshots and all post-root actions,
positions, rewards, deaths and completion flags. Source file hashes are checked.

## Endpoints and interpretation

Local forward escape means reaching at least 64 pixels beyond root x at the
end of the horizon without death, or completing the level during the horizon.
Also report maximum displacement, deaths, final displacement and local returns.
Crossing the threshold and then dying is not a successful escape. Save every
branch outcome; never choose the best seed or omit adverse interventions.

These are local intervention results, not full-level completion rates, a
deployable rescue detector, or a confirmation of human-level performance.
Interventions alter subsequent states and observations as well as action history;
a rescue establishes sufficiency at those tested roots, not a unique internal
cause. Root screenshots and original priors/visits can help identify obstacles.
Further held-out testing is required before choosing a pulse as a controller.

## Execution

Run `scripts/submit_stall_pulses.sh prepare-smoke OUT` on one GPU. Freeze source,
checkpoint copies and case plan, run four targeted checks, then all five arms
on the first case of each level with a 16-decision smoke horizon. Smoke results
are validation only. Full per-level jobs use the same frozen plan, all retained
cases and the 128-decision horizon, with afterok dependencies on validation.
All replay, inference and tests run on compute nodes; artifacts live in `/projects`.

## Submitted and validated

Output: `outputs/controller_policy/stall-pulses-v1-20260911/`.

| Task | Slurm job | Status at initial audit |
|---|---:|---|
| Preparation, tests and smoke checks | 6459314 | Completed 0:0, 1m10s |
| Full Level1-1 intervention test | 6459327 | Submitted after validation |
| Full Level6-1 intervention test | 6459328 | Submitted after validation |

All four targeted tests passed. All ten smoke branches completed and verified
reset provenance, replayed rewards/positions and exact saved root observations.
The fixed selection retained **19 of 19** Level1-1 timeouts, with no exclusions.
The paired Level6-1 cohort originally completed **17/19** episodes; its two
failures are retained. Full jobs schedule **95 branches per level**, five per
root, using 128 decisions each. Smoke results use just 16 decisions from the
first root and must not be interpreted as the full local-escape result.

The first Level1-1 root screenshot shows Mario at the base of a staircase;
this observation identifies one concrete obstacle to inspect, not the cause of
every stall. Full root/end screenshots and JSON branches are retained for all
cases. Read the actual full summaries' `complete` flags before reporting results.
