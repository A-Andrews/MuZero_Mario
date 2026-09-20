# Actual searched paths and observed-state leaf values

Requested 2026-09-19 following `VALUE_COMMITMENT_RESULTS_V1.md`. This round tests
whether replacing imagined leaf values changes MCTS decisions and subsequent
outcomes. It does not change trained weights or propose a deployable emulator
lookahead controller.

2026-09-20: all 1,120 gameplay branches finished. Reporting collision repaired;
outcome audit 6721905 passed. Search metadata recovery is complete for all 56
roots; final summary 6721927 passed. See [results](SEARCH_LEAF_VALUES_RESULTS_V1.md).

Runner: `scripts/diagnose_search_leaf_values.py`; submit wrapper:
`scripts/submit_search_leaf_values.sh`; frozen template:
`docs/search_leaf_values_v1.json`. Output:
`outputs/controller_policy/search-leaf-values-v1b-20260919/`.

## Fixed scope and paired arms

Reuse all 56 distinct states from the completed value/commitment plan, without
selection based on this new intervention: 46 roots from 23 failed source
episodes and ten reference roots from five successful source episodes.
Levels 1-1 and 6-1 only; greedy and stall-sampled source controllers remain
separate. Successful 6-1 references remain intact-control comparisons.

Use the same ten predetermined branch seeds, checkpoints, 50 simulations,
leaf batch four, no root noise, controller temperatures, restored stall-monitor
state and original total allowance of 2,000 decisions. The three conditions:

1. Normal search: reuse audited full baseline outcomes; reproduce its first
   16 decisions with observation-only instrumentation and exact search/RNG checks.
2. Replace leaf values for the next **one** decision, then resume normal search.
3. Replace leaf values for the next **eight** decisions, then resume normal search.

There are 1,120 new full branches (56 × ten seeds × two interventions), plus
560 reused baseline outcomes. Ten repeats from a state are conditional search
repetitions, not independent gameplay starts. No action holding is applied.

## Intervention and instrumentation

Use the unchanged production MCTS implementation. A temporary diagnostic hook
intercepts `Node.backup` at each evaluated leaf and walks its parents to recover
the actual selected action path. The initial root backup remains unchanged.
For each of the 50 simulation backups, record the path/depth, original imagined
value, observed-state value, backed-up value, predicted immediate reward,
observation hash and terminal metadata. Duplicate leaf selections within a
batch are retained as separate backups; unique path counts are also reported.

In the substitution arm, replace only the scalar leaf value entering backup.
Keep imagined hidden states, root and child policy priors, predicted rewards,
search budget, selection rule and batch handling unchanged. Subsequent paths
may differ because corrected backups alter search selection: that is part of
the tested effect. This is not an evaluation on fixed eight-action prefixes.

At every modified decision, run a normal shadow search from exactly the same
real observation and starting search RNG state. Discard its RNG advancement
before running the intervention. This compares action/visit changes on matched
states, even after the intervention has diverged from the original trajectory.
Observation-only instrumentation must reproduce plain search's action, visits,
root Q and ending RNG exactly. It is checked at each instrumented decision.

## Obtaining actual observations without incomplete snapshots

A separate CPU emulator process resets and replays the original recorded
actions to the saved diagnostic root, verifying reset provenance, rewards,
positions and the saved root observation. It then executes the current real
branch prefix followed by the requested search path. The parent process feeds
the resulting observation through the frozen representation/value network in
eval mode. Thus every replacement value uses the same network as ordinary
search; it is not a Monte Carlo return or ground-truth value.

The worker uses complete replay, not an emulator-only snapshot that might omit
frame-stack/reward-wrapper state. It is isolated from the live branch emulator.
Before each replay the worker restores the original random-start generator
seed, so reusing an emulator is equivalent to creating a fresh seeded one.
Identical full action paths share cached values across seeds/arms within the
same root. The exact live-root observation hash is checked against replay at
each instrumented decision. Observation queries preserve the search RNG.
Actual rewards are recorded for audit but never replace predicted search rewards.

If a path terminates, its replacement value is zero; record the first terminal
depth measured from the saved diagnostic root (including the actual executed
prefix). Stop emulator execution at that terminal. Search itself is not pruned
or given real rewards: imagined post-terminal rewards may still be backed up.
Report terminal-touching paths separately when interpreting the intervention,
because terminal knowledge is additional privileged information. No claim of
fully correct model-based planning follows from this value-only substitution.

## Checks and outputs

Focused tests cover exact observation-only equivalence at leaf batches 1/4 and
temperatures 0/.25, query RNG isolation, path ordering, changed decisions under
known synthetic leaf values, preservation of priors/rewards, terminal handling,
and hook cleanup after failure. Existing MCTS and value/commitment tests remain
part of validation. Production source hashes must match the previous experiment.

Freeze source/checkpoints/plan, then smoke-test roots 0, 2, 3, 18, 35, 37 and 40:
both levels, every source group, successful references, and the late 6-1 obstacle.
Smoke uses one seed and 16 real decisions; it is not performance evidence.
Full arrays start only after successful validation. Retain per-seed JSON search
records and paired NPZ branch traces, a per-root observation-value cache and
provenance. Final audit requires every planned seed/root/arm and checks trace
hashes before aggregating paired rescues and lost successes.

Primary readouts: actual search depth and value discrepancies; changed first
action and visits; paired completion/death/timeout changes; local survival and
progress; harms on successful controls. Failures remain grouped by signature
and source controller. Improvement would implicate imagined-state valuation in
the search behaviour at these states, without isolating dynamics from
representation/value-head interactions or establishing whole-level performance.
No improvement would leave real-observation value errors, priors, predicted
rewards and search allocation as possible limitations.

## Validation correction and jobs

- Preflight **6702295**: all 21 focused/existing tests passed.
- Initial emulator validation **6702313** stopped at an initial-observation
  mismatch: repeated resets on the reused worker advanced its random-start
  generator. No full experiment ran. Dependent arrays 6702318/6702319 and audit
  6702329 were automatically cancelled. Preserve that failed snapshot as evidence.
- Corrected fresh snapshot **v1b**, validation **6702349**: reset the worker's
  random-start generator from the original environment seed for every replay;
  add a regression test for repeated queries. The seven planned smoke roots
  include the late 6-1 obstacle and successful references.
- Corrected validation has passed all **22 tests** and progressed through the
  first single-decision substitution smoke probe with replay/search checks intact.
  Full smoke validation is still running at submission; no full outcomes yet.
- **6702399**: corrected 6-1 array, roots 35–55, at most two concurrent GPUs.
- **6702400**: corrected 1-1 array, roots 0–34, at most two concurrent GPUs.
- **6702401**: corrected CPU-only final audit after both arrays succeed.

Full arrays have `afterok` dependencies and cancellation on invalid dependencies;
submission alone does not mean validation or the experiment is complete.
