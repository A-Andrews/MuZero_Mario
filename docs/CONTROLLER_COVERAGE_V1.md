# Remaining-level stochastic evaluation, 2026-09-11

This records the historical 22-level experiment. Current project scope excludes
levels without human data; see `HUMAN_LEVEL_SCOPE.md`. The active graph now has
21 rows after dropping 2-2. Future submissions use the separate
`controller_coverage_human_v1.json` template (17 extension levels).

**Completed:** all 18 array tasks and figure job 6459441 succeeded; all 1,080
new episodes were collected without recorded infrastructure errors. See
[CONTROLLER_INVESTIGATION_RESULTS_20260911.md](CONTROLLER_INVESTIGATION_RESULTS_20260911.md).

The user requested stochastic evaluations for all remaining levels. This expands
the original 22-level performance benchmark to paired greedy and sampling-only
MCTS results everywhere, using the four already completed development levels
and **18 new levels**. It does not select levels based on failure or success.

New levels: 1-2, 1-3, 2-1, 2-2, 2-3, 3-1, 3-3, 4-1, 4-2, 4-3, 5-1, 5-2,
6-2, 6-3, 7-1, 7-3, 8-2, 8-3. The completed 1-1, 3-2, 6-1 and 8-1 results
are reused with their original sample sizes. Level2-2 has no human band, but
is included in model evaluation. Level5-3 is outside the original figure and
has no best.pt in either specialist run; only latest.pt exists. No substitute
checkpoint or additional run is silently introduced for it.

## Fixed protocol

Use the same 30 development environment/search seed pairs and reset settings as
v1b. Compare two conditions, interleaved by episode index:

- Greedy: leaf batch 4, 50 simulations, root epsilon 0, temperature 0.
- Sampling-only: identical settings, temperature .25.

Each attempt has the original 2000-decision cap. All failures and timeouts count.
Infrastructure errors stop a task and preserve its error; resume must use the
same scheduled seeds. Do not tune temperature, replace failed seeds or label
development outcomes as held-out confirmation. There are **1,080 new episodes**
(18 levels × 2 controllers × 30 trials). The reserved 100-trial seed list stays
unused by this extension.

Run selections come from the original performance figure, including its selected
imitation specialists. Freeze the current best.pt from each selected run before
evaluation, with source and checkpoint SHA256 provenance. The historical run's
training step is retained as provenance, but is not assumed to identify a current
file that may have changed. The paired fresh greedy arm supplies a comparison
on exactly the checkpoint used for sampling.

## Control and validation

Include Level6-1 explicitly in the manifest and copy its original frozen
checkpoint with an expected SHA check. Reproduce the first development greedy
and sampled trials, comparing exact actions, priors, visits, rewards and x traces
against the completed control data. Reuse its full original n=30 per controller;
these smoke repetitions do not add independent trials. Also smoke-test both
controllers on the selected Level1-3 imitation checkpoint.

`coverage_extension` is an explicit development profile in the evaluator. It
requires both declared controllers, all 30 trials, the successful control, and
the same immutable resume/accounting checks as the original protocol. The
original five-arm profile and 100-trial confirmation profile remain strict.

## Compute execution and figure

Preparation, tests, checkpoint reads and rollouts run on Slurm compute nodes.
Freeze artifacts under `outputs/controller_policy/coverage-v1-20260911/` on
`/projects`. Submit an 18-task array with **at most four GPU tasks running
concurrently**, one GPU per task, after successful validation. Each task has a
six-hour ceiling. The original confirmation and stall-intervention jobs continue
independently.

After all array tasks succeed, generate
`images/human_vs_agent_runthrough_all_levels_stochastic.{pdf,png,json}` from
the extension plus the four completed original development levels. Every model
row will then have 30 attempts per controller. The plotting reader refuses
incomplete summaries, checks identities/counts, and preserves the original human
time bands. This figure remains development-only; the separate confirmation
figure retains its own labels and denominators.

## Submission record

| Task | Job |
|---|---:|
| Frozen preparation, tests and smoke checks | 6459439 |
| 18-level evaluation array, concurrency 4 | 6459440, indices 0–17 |
| Completed all-level stochastic figure | 6459441 |

The array depends on successful validation; the figure depends on successful
completion of the entire array. Invalid dependencies cancel the dependent job.
Validation completed successfully: all 16 targeted tests passed, all four smoke
episodes finished, and exact Level6-1 actions, priors, visits, rewards and x
traces matched the original experiment. `control_validation.json` has
`passed=true`. The reference-only 6-1 smoke does not increment its reused
30-trial count. Check the array status and per-level summary `complete` fields
before reporting the full evaluation results; smoke outcomes are not estimates.
